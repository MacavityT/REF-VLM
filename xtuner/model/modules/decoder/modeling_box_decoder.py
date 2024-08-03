# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_box_decoder import BoxDecoderConfig

from transformers.models.detr import DetrConfig
from transformers.models.detr.modeling_detr import (
    DetrDecoder, 
    DetrMLPPredictionHead,
    generalized_box_iou
)
from transformers.image_transforms import center_to_corners_format

class BoxDecoderModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = BoxDecoderConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: BoxDecoderConfig):
        super().__init__(config)
        self.config = config

        # Create projection layer
        self.query_position_embeddings = nn.Embedding(
            config.num_queries, 
            config.d_model
        )
        decoder_config = DetrConfig(
            decoder_layers=config.decoder_layers,
            decoder_ffn_dim=config.decoder_ffn_dim,
            decoder_attention_heads=config.decoder_attention_heads,
            decoder_layerdrop=config.decoder_layerdrop,
            activation_function=config.activation_function,
            d_model=config.d_model,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            init_std=config.init_std,
            init_xavier_std=config.init_xavier_std,
        )
        self.decoder = DetrDecoder(decoder_config)
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, 
            hidden_dim=config.d_model, 
            output_dim=4, 
            num_layers=3
        )
        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BoxDecoderModel):
            module.gradient_checkpointing = value
            
    def loss_boxes(self, preds, targets, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        num_boxes = len(targets)
        assert len(preds) == len(targets)

        loss_bbox = nn.functional.l1_loss(preds, targets, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(preds), center_to_corners_format(targets))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def forward(self, x, labels):
        batch_size = x.shape[0]
        hidden_states = x

        attention_mask = None

        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=attention_mask,
            object_queries=None, # position embedding for pixel values
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=None, # mask for pixel values
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs[0]
        boxes_pred_logits = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict = None, None
        if labels is not None:
            weight_dict = {
                "loss_bbox": self.config.bbox_loss_coefficient,
                "loss_giou": self.config.giou_loss_coefficient}
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return boxes_pred_logits, loss
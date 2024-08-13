# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .modeling_decoder import DecoderModel
from .configuration_box_decoder import BoxDecoderConfig

from transformers.models.detr import DetrConfig
from transformers.models.detr.modeling_detr import (
    DetrDecoder,
    DetrDecoderLayer,
    DetrMLPPredictionHead,
    generalized_box_iou
)
from transformers.image_transforms import center_to_corners_format

class BoxDecoderModel(DecoderModel):
    config_class = BoxDecoderConfig

    def __init__(self, config: BoxDecoderConfig):
        super().__init__(config)
        self.config = config

        # Create projection layer
        self.in_proj_visual_feats = nn.Linear(self.in_channels, config.d_model)
        self.in_proj_queries = nn.Linear(config.quries_input_dim, config.d_model)
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
        )
        self.decoder = DetrDecoder(decoder_config)
        self.predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, 
            hidden_dim=config.d_model, 
            output_dim=4, 
            num_layers=3
        )
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def compute_loss_box(self, preds, targets):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (x1, y1, x2, y2), normalized by the image size.
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

    def forward(self, 
        visual_hidden_states,
        ref_hidden_states,
        visual_mask=None,
        ref_mask=None,
        metas=None,
        mode='loss'
    ):
        # prepare visual hidden states
        visual_hidden_states = self.transform_visual_inputs(visual_hidden_states)
        visual_hidden_states = self.in_proj_visual_feats(visual_hidden_states)
        visual_position_embedding, visual_flatten_mask = self.visual_position_encoding(
            visual_hidden_states, 
            visual_mask
        )

        # prepare learnable queries
        batch_size = ref_hidden_states.shape[0]
        ref_hidden_states = self.in_proj_queries(ref_hidden_states)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        assert ref_hidden_states.shape == query_position_embeddings.shape

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=ref_hidden_states,
            attention_mask=None, # could be ref_mask, still with some bugs
            object_queries=visual_position_embedding,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=visual_hidden_states,
            encoder_attention_mask=visual_flatten_mask, # mask for pixel values
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs[0]
        pred_boxes = self.predictor(sequence_output).sigmoid()

        decode_units = metas.get('decode_units', None)
        if decode_units is not None:
            batch_remove_mask = [unit != 'box' for unit in decode_units]  
            # remove batch which task not box deocde 
            condition = torch.zeros_like(ref_mask).to(torch.bool)
            condition[batch_remove_mask, :] = True 
            ref_mask = ref_mask.masked_fill(condition, False)
        pred_boxes = pred_boxes[ref_mask, :]

        loss = None
        if mode == 'loss':
            try:
                target_boxes = self.get_unit_labels(metas, ref_mask, 'box')
            except:
                from xtuner.model.utils import save_wrong_data
                save_wrong_data(f"wrong_ref",metas)
            if ref_mask.sum() > 0 and target_boxes is not None:
                target_boxes = torch.stack(
                    target_boxes).to(pred_boxes.device).to(pred_boxes.dtype)
                loss_dict = self.compute_loss_box(pred_boxes, target_boxes)
                weight_dict = {
                    "loss_bbox": self.config.bbox_loss_coefficient,
                    "loss_giou": self.config.giou_loss_coefficient}
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            else:
                loss = pred_boxes.sum() * 0.0

        return dict(
            loss=loss,
            preds=pred_boxes
        )
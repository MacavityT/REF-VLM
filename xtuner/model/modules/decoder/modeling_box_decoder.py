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
    DetrMLPPredictionHead,
    generalized_box_iou
)
from transformers.image_transforms import center_to_corners_format
from transformers.utils import requires_backends, is_scipy_available
from xtuner.model.utils import save_wrong_data

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

class BoxDecoderGroupHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Box Matcher can't be 0")

    @torch.no_grad()
    def forward(self, pred_boxes, target_boxes, target_slices):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # bf16 and fp16 not supported in linear_sum_assignment function
        if pred_boxes.dtype in [torch.float16, torch.bfloat16]:
            pred_boxes = pred_boxes.to(torch.float32)
        if target_boxes.dtype in [torch.float16, torch.bfloat16]:
            target_boxes = target_boxes.to(torch.float32)
        
        # pred_boxes shape = [num_boxes, 4]
        assert len(pred_boxes) == len(target_boxes), "pred_boxes num must equal to target_boxes num"
        assert len(pred_boxes) == sum(target_slices), "pred_boxes num must equal to sum of target_slices"
        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(pred_boxes, target_boxes, p=1)
        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(pred_boxes), center_to_corners_format(target_boxes))

        # Final cost matrix, shape of [pred_num, target_num]
        cost_matrix = self.bbox_cost * bbox_cost + self.giou_cost * giou_cost

        # re-organization 
        cost_matrix = cost_matrix.cpu()
        start = 0
        indices = []
        for slices in target_slices:
            end = start + slices
            matrix_slice = cost_matrix[start:end, start:end]
            indice = linear_sum_assignment(matrix_slice)
            indice = [start + idx for idx in indice]
            indices.append(indice)
            start = end
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class BoxDecoderLoss(nn.Module):

    def __init__(self, matcher=None):
        super().__init__()
        self.matcher = matcher

    def _get_permutation_idx(self, indices):
        # permute predictions following indices
        source_idx = []
        target_idx = []
        for (source, target) in indices:
            source_idx.extend(source)
            target_idx.extend(target)
        source_idx = torch.stack(source_idx)
        target_idx = torch.stack(target_idx)
        return source_idx, target_idx

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
        loss_bbox = nn.functional.l1_loss(preds, targets, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(preds), center_to_corners_format(targets))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def forward(self, pred_boxes, target_boxes, target_slices):
        if self.matcher is not None:
            indices = self.matcher(pred_boxes, target_boxes, target_slices)
            source_idx, target_idx = self._get_permutation_idx(indices)
            pred_boxes = pred_boxes[source_idx] # actually do nothing, matcher only permuted the target idx
            target_boxes = target_boxes[target_idx]
        loss_dict = self.compute_loss_box(pred_boxes, target_boxes)
        return loss_dict

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

        if config.use_group_matcher:
            matcher = BoxDecoderGroupHungarianMatcher(
                bbox_cost=config.bbox_loss_coefficient,
                giou_cost=config.giou_loss_coefficient
            )
        else:
            matcher = None
        self.criteria = BoxDecoderLoss(matcher)
        # Initialize weights and apply final processing
        self.post_init()

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
        ref_hidden_states = ref_hidden_states[:, :self.config.num_queries, :]
        ref_mask = ref_mask[:, :self.config.num_queries]

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
                target_slices = self.get_label_slices(metas, ref_mask)
            
                if ref_mask.sum() > 0 and target_boxes is not None:
                    target_boxes = torch.stack(
                        target_boxes).to(pred_boxes.device).to(pred_boxes.dtype)
                    loss_dict = self.criteria(pred_boxes, target_boxes, target_slices)
                    weight_dict = {
                        "loss_bbox": self.config.bbox_loss_coefficient,
                        "loss_giou": self.config.giou_loss_coefficient}
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                else:
                    loss = pred_boxes.sum() * 0.0

            except Exception as e:
                print(e)
                metas['type'] = 'box'
                metas['ref_mask_filter'] = ref_mask
                save_wrong_data(f"wrong_ref_match", metas)
                raise ValueError('Error in get_unit_labels/seqs process, type = box')
        return dict(
            loss=loss,
            preds=pred_boxes
        )
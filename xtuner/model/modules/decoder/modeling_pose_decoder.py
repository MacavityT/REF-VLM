# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .modeling_decoder import DecoderModel
from .modeling_box_decoder import BoxDecoderModel, BoxDecoderLoss, BoxDecoderGroupHungarianMatcher
from .configuration_pose_decoder import PoseDecoderConfig, KeypointDecoderConfig

from transformers.models.detr import DetrConfig
from transformers.models.detr.modeling_detr import (
    DetrDecoder,
    DetrMLPPredictionHead,
)
from transformers.image_transforms import center_to_corners_format
from xtuner.model.utils import save_wrong_data

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class KeypointDecoderLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices = indices[0]
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def compute_loss_oks(self, 
        kpt_preds,
        kpt_gts,
        kpt_valids=None,
        kpt_areas=None,
        linear=False,
        num_keypoints=None,
        eps=1e-6
    ):
        """Oks loss.
        Computing the oks loss between a set of predicted poses and target poses.
        The loss is calculated as negative log of oks.
        Args:
            pred (torch.Tensor): Predicted poses of format (x1, y1, x2, y2, ...),
                shape (n, 2K).
            target (torch.Tensor): Corresponding gt poses, shape (n, 2K).
            linear (bool, optional): If True, use linear scale of loss instead of
                log scale. Default: False.
            eps (float): Eps to avoid log(0).
        Return:
            torch.Tensor: Loss tensor.
        """

        if num_keypoints == 17:
            sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

        sigmas = kpt_preds.new_tensor(sigmas)
        variances = (sigmas * 2)**2

        assert kpt_preds.size(0) == kpt_gts.size(0)
        kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
        kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)

        squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
            (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
        # import pdb
        # pdb.set_trace()
        # assert (kpt_valids.sum(-1) > 0).all()
        squared_distance0 = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
        squared_distance1 = torch.exp(-squared_distance0)
        squared_distance1 = squared_distance1 * kpt_valids
        oks = squared_distance1.sum(dim=1) / (kpt_valids.sum(dim=1)+1e-6)
        oks = oks.clamp(min=eps)
        
        if linear:
            loss = 1 - oks
        else:
            loss = -oks.log()
        return loss

    def compute_loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the keypoints
        """
        indices = indices[0]
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv

        if len(src_keypoints) == 0:
            device = outputs["pred_logits"].device
            losses = {
                'loss_keypoints': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
                'loss_oks': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
            }
            return losses
        Z_pred = src_keypoints[:, 0:(self.num_body_points*2)]
        V_pred = src_keypoints[:, (self.num_body_points*2):]
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points*2)]
        V_gt: torch.Tensor = targets_keypoints[:, (self.num_body_points*2):]
        oks_loss=self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        losses = {}
        losses['loss_keypoints'] = pose_loss.sum() / num_boxes        
        losses['loss_oks'] = oks_loss.sum() / num_boxes
        return losses

    def forward(self, pred_boxes, target_boxes, target_slices):
        if self.matcher is not None:
            indices = self.matcher(pred_boxes, target_boxes, target_slices)
            source_idx, target_idx = self._get_permutation_idx(indices)
            pred_boxes = pred_boxes[source_idx] # actually do nothing, matcher only permuted the target idx
            target_boxes = target_boxes[target_idx]
        
        loss_dict = dict()
        loss_dict_box = self.compute_loss_box(pred_boxes, target_boxes)
        loss_dict_kpt = self.compute_loss_keypoints()
        loss_dict.update(loss_dict_box)
        loss_dict.update(loss_dict_kpt)
        return loss_dict

class KeypointDecoderModel(DecoderModel):
    def __init__(self, config: KeypointDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # Create projection layer
        self.in_proj_visual_feats = nn.Linear(self.in_channels, config.d_model)
        self.in_proj_queries = nn.Linear(config.quries_input_dim, config.d_model)
        
        self.register_buffer(
            "kpt_ids", 
            torch.arange(config.num_body_points).unsqueeze(1), 
            persistent=False
            )
        self.kpt_embedding = nn.Embedding(config.num_body_points, config.d_model)
        num_combine_queries = int((config.num_body_points + 1) * config.num_queries)
        self.query_position_embeddings = nn.Embedding(
            num_combine_queries,
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
        self.box_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, 
            hidden_dim=config.d_model, 
            output_dim=2, 
            num_layers=3
        )
        self.kpt_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, 
            hidden_dim=config.d_model, 
            output_dim=config.num_body_points, 
            num_layers=3
        )
        self.kpt_classifier = nn.Linear(config.d_model, 2)

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
        if isinstance(visual_hidden_states, torch.Tensor):
             visual_hidden_states = self.bchw2blc(visual_hidden_states)
        else:
            visual_hidden_states = [self.bchw2blc(hidden_states) for hidden_states in visual_hidden_states]
        visual_hidden_states = self.in_proj_visual_feats(visual_hidden_states)
        visual_position_embedding, visual_flatten_mask = self.visual_position_encoding(
            visual_hidden_states, 
            visual_mask
        )

        # prepare learnable queries
        batch_size = ref_hidden_states.shape[0]
        ref_hidden_states = self.in_proj_queries(ref_hidden_states)

        ref_hidden_states_expand = ref_hidden_states.repeat(1, self.config.num_body_points + 1, 1)
        kpt_queries_init = torch.cat([torch.zeros((1, 1, self.config.d_model)), 
                                      self.kpt_embedding(self.kpt_ids)], dim=1)
        ref_hidden_states_expand = ref_hidden_states_expand + kpt_queries_init
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        assert ref_hidden_states_expand.shape == query_position_embeddings.shape

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=ref_hidden_states_expand,
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
        
        box_queries_logits = []
        kpt_queries_logits = []
        batch, ref_len, _ = ref_hidden_states.shape
        dim = sequence_output.shape[-1]
        for idx in range(ref_len):
            box_idx = idx * (self.config.num_body_points + 1)
            kpt_start = box_idx + 1
            kpt_end = kpt_start + self.config.num_body_points

            box_queries_logits.append(sequence_output[:, box_idx, :])
            kpt_queries_logits.append(
                sequence_output[:, kpt_start:kpt_end, :].view(
                    batch, 1, self.config.num_body_points, dim
                )
            )
        box_queries_logits = torch.cat(box_queries_logits, dim=1)
        kpt_queries_logits = torch.cat(kpt_queries_logits, dim=1)

        box_queries_logits = self.box_predictor(box_queries_logits).sigmoid()
        kpt_queries_logits = self.box_predictor(kpt_queries_logits).sigmoid()
        kpt_cls_logits = self.kpt_classifier(kpt_queries_logits)
        
        return dict(
            hidden_states = sequence_output,
            box_logits = box_queries_logits,
            kpt_logits = kpt_queries_logits,
            kpt_cls_logits = kpt_cls_logits
        )

class PoseDecoderModel(DecoderModel):
    config_class = PoseDecoderConfig

    def __init__(self, config: PoseDecoderConfig):
        super().__init__(config)
        self.config = config
        self.box_decoder = BoxDecoderModel(config.box_config)
        self.kpt_decoder = KeypointDecoderModel(config.keypoint_config)

        self.box_criteria = self.box_decoder.criteria
        self.kpt_criteria = KeypointDecoderLoss(self.box_decoder.criteria)
        self.post_init()

    def forward(self, 
        visual_hidden_states,
        ref_hidden_states,
        visual_mask=None,
        ref_mask=None,
        metas=None,
        mode='loss'
    ):
        box_outputs = self.box_decoder(
            visual_hidden_states=visual_hidden_states,
            ref_hidden_states=ref_hidden_states,
            visual_mask=visual_mask,
            ref_mask=ref_mask,
            metas=metas,
            mode='tensor'
        )

        if box_outputs is None:
            return box_outputs

        kpt_outputs = self.kpt_decoder(
            visual_hidden_states=visual_hidden_states,
            ref_hidden_states=box_outputs['ref_hidden_states'],
            visual_mask=visual_mask,
            ref_mask=box_outputs['ref_mask'],
            metas=metas,
            mode=mode
        )

        # get logits
        boxes_queries_logits1 = box_outputs['box_logits']
        boxes_queries_logits2 = kpt_outputs['box_logits']
        kpts_queries_logits = kpt_outputs['kpt_logits']
        kpts_cls_logits = kpt_outputs['kpt_cls_logits']
        
        loss = None
        if mode == 'loss':
            try:
                target_boxes = self.get_unit_labels(metas, ref_mask, 'box')
                target_kpts = self.get_unit_labels(metas, ref_mask, 'kpt')
                target_slices = self.get_label_slices(metas, ref_mask)
            except Exception as e:
                print(e)
                metas['type'] = 'box'
                metas['ref_mask_filter'] = ref_mask
                save_wrong_data(f"wrong_ref_match", metas)
                raise ValueError('Error in get_unit_labels/seqs process, type = pose')
            
            pred_boxes1 = boxes_queries_logits1[ref_mask, :] # [b, n, 4]
            pred_boxes2 = boxes_queries_logits2[ref_mask, :] # [b, n, 4]
            pred_kpts = kpts_queries_logits[ref_mask, ...] # [b, n, num_body_points, 2]

            if ref_mask.sum() > 0 and target_boxes is not None:
                target_boxes = torch.stack(
                    target_boxes).to(pred_boxes.device).to(pred_boxes.dtype)
                loss_dict = self.criteria(pred_boxes, target_boxes, target_slices)
                weight_dict = {
                    "loss_bbox": self.config.bbox_loss_coefficient,
                    "loss_giou": self.config.giou_loss_coefficient}
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            else:
                loss = pred_boxes.sum() * 0.0 + pred_kpts.sum() * 0.0
        else:
            assert mode == 'predict'
            boxes_queries_logits = self.predictor(sequence_output).sigmoid()
            pred_boxes = []
            for queries_logits, mask in zip(boxes_queries_logits, ref_mask):
                boxes = queries_logits[mask, :]
                pred_boxes.append(boxes)

        return dict(
            loss=loss,
            preds=pred_boxes,
        )
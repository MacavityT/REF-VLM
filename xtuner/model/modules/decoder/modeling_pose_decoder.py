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

class PoseDecoderLoss(BoxDecoderLoss):

    def __init__(self, matcher=None, num_body_points=17, eps=1e-6):
        super().__init__(matcher)
        self.num_body_points = num_body_points
        if num_body_points == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_body_points == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_body_points}')
        self.eps = eps

    # removed logging parameter, which was part of the original implementation
    def compute_loss_labels(self, preds, targets):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        origin_type = preds.dtype
        preds = preds.view(-1, preds.shape[-1]).to(torch.float32)
        targets = targets.view(-1).to(torch.int64)
        loss_ce = nn.functional.cross_entropy(preds, targets)
        loss_ce.to(origin_type)
        losses = {"loss_ce": loss_ce}
        return losses

    def compute_loss_oks(self, pred_keypoints, gt_keypoints, visible, area):
        """
        :param pred_keypoints: 预测关键点的坐标，形状为 (batch_size, num_keypoints, 2)
        :param gt_keypoints: 真实关键点的坐标，形状为 (batch_size, num_keypoints, 2)
        :param visible: 关键点的可见性，形状为 (batch_size, num_keypoints)，1为可见，0为不可见
        :param area: 对象的面积（例如包围框的面积），形状为 (batch_size, 1)
        :return: 计算出的OKS损失
        """
        # 计算预测关键点和真实关键点之间的欧式距离
        distance = torch.sqrt(((pred_keypoints - gt_keypoints) ** 2).sum(dim=-1))

        # 计算归一化的距离平方
        sigmas = pred_keypoints.new_tensor(self.sigmas)
        area = area.unsqueeze(-1)  # 将area扩展到与关键点匹配的维度
        sigma_term = 2 * (sigmas ** 2) * (area + self.eps)
        oks_term = torch.exp(- (distance ** 2) / sigma_term)

        # 只计算可见的关键点
        oks_term = oks_term * visible.float()

        # 计算OKS，防止除以0
        oks_loss = 1.0 - oks_term.sum(dim=-1) / (visible.sum(dim=-1).float() + self.eps)
        oks_loss = torch.clamp(oks_loss, 0, 1)

        losses = {}
        losses['loss_oks'] = oks_loss.mean() 
        return losses

    def compute_loss_keypoints(self, pred_kpts, target_kpt, valid_kpts):
        """Compute the losses related to the keypoints
        """
        num_boxes = len(target_kpt)
        pose_loss = F.l1_loss(pred_kpts, target_kpt, reduction='none')
        pose_loss = pose_loss[valid_kpts, :]  # ignore not visible keypoints
        losses = {}
        losses['loss_keypoints'] = pose_loss.sum() / num_boxes        
        return losses

    def forward(self, pred_boxes, target_boxes, pred_kpts, pred_kpts_cls, target_kpts, target_slices):
        if self.matcher is not None:
            indices = self.matcher(pred_boxes, target_boxes, target_slices)
            source_idx, target_idx = self._get_permutation_idx(indices)
            pred_boxes = pred_boxes[source_idx] # actually do nothing, matcher only permuted the target idx
            pred_kpts = pred_kpts[source_idx]
            pred_kpts_cls = pred_kpts_cls[source_idx]
            target_boxes = target_boxes[target_idx]
            target_kpts = target_kpts[target_idx]

        # target process
        target_kpts_coord = target_kpts[..., :-1]
        target_kpts_cls = target_kpts[..., -1]
        valid_kpts = (target_kpts_cls > 0).to(torch.bool).to(target_kpts_cls.device)
        kpt_areas = target_boxes[:, 2] * target_boxes[:, 3]
        
        loss_dict = dict()
        loss_dict.update(self.compute_loss_box(pred_boxes, target_boxes))
        loss_dict.update(self.compute_loss_keypoints(pred_kpts, target_kpts_coord, valid_kpts))
        loss_dict.update(self.compute_loss_labels(pred_kpts_cls, target_kpts_cls))
        loss_dict.update(self.compute_loss_oks(pred_kpts, target_kpts_coord, valid_kpts, kpt_areas))
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
            torch.arange(config.num_body_points).unsqueeze(0), 
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
            output_dim=4, 
            num_layers=3
        )
        self.kpt_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, 
            hidden_dim=config.d_model, 
            output_dim=2, 
            num_layers=3
        )
        self.kpt_classifier = nn.Linear(config.d_model, 3)

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
        empty_tensor = torch.zeros((1, 1, self.config.d_model)).to(ref_hidden_states.dtype).to(ref_hidden_states.device)
        kpt_queries_init = torch.cat([empty_tensor, self.kpt_embedding(self.kpt_ids)], dim=1)
        kpt_queries_init = kpt_queries_init.repeat(1, self.config.num_queries, 1)
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
        for idx in range(ref_hidden_states.shape[1]):
            box_idx = idx * (self.config.num_body_points + 1)
            kpt_start = box_idx + 1
            kpt_end = kpt_start + self.config.num_body_points

            box_queries_logits.append(sequence_output[:, box_idx, :])
            kpt_queries_logits.append(sequence_output[:, kpt_start:kpt_end, :])
        box_queries_logits = torch.stack(box_queries_logits, dim=1)
        kpt_queries_logits = torch.stack(kpt_queries_logits, dim=1)

        box_logits = self.box_predictor(box_queries_logits).sigmoid()
        kpt_logits = self.kpt_predictor(kpt_queries_logits).sigmoid()
        kpt_cls_logits = self.kpt_classifier(kpt_queries_logits)
        
        return dict(
            hidden_states = sequence_output,
            box_logits = box_logits,
            kpt_logits = kpt_logits,
            kpt_cls_logits = kpt_cls_logits
        )

class PoseDecoderModel(PreTrainedModel):
    config_class = PoseDecoderConfig

    def __init__(self, config: PoseDecoderConfig):
        super().__init__(config)
        self.config = config
        self.box_decoder = BoxDecoderModel(config.box_config)
        self.kpt_decoder = KeypointDecoderModel(config.keypoint_config)
        if config.use_group_matcher:
            matcher = BoxDecoderGroupHungarianMatcher(
                bbox_cost=config.box_config.bbox_loss_coefficient,
                giou_cost=config.box_config.giou_loss_coefficient
            )
        else:
            matcher = None
        self.criteria = PoseDecoderLoss(matcher)
        self.post_init()

    def get_label_slices(self, metas, ref_mask):
        decode_seqs = metas.get('decode_seqs', None)
        if decode_seqs is None:
            return None
        
        target_slices =[]
        for seqs in decode_seqs:
            if seqs is None:
                target_slices.append([])
            else:
                u_num = [len(seq) for seq in seqs]
                target_slices.append(u_num)
        
        slice_num = sum([len(items) for items in target_slices])
        if slice_num == 0:
            return None

        # get used seqs in batch data (sequence might be cutoff and remove some seqs)
        target_slices_trim = []
        for batch_idx, mask in enumerate(ref_mask):
            ref_num = mask.sum().cpu().item()
            if ref_num == 0: continue

            unit_num = target_slices[batch_idx]
            assert sum(unit_num) >= ref_num
            diff = sum(unit_num) - ref_num
            while diff > 0:
                cur_diff = diff
                diff -= min(unit_num[-1], cur_diff)
                unit_num[-1] -= min(unit_num[-1], cur_diff)
                if unit_num[-1] == 0: unit_num.pop()
            target_slices_trim.extend(unit_num)
        return target_slices_trim

    def get_unit_labels(self, metas, ref_mask, type):
        decode_labels = metas.get('decode_labels', None)
        if decode_labels is None:
            return None
        
        target_labels =[]
        for labels in decode_labels:
            if labels is None: 
                target_labels.append([])
            else:
                unit_labels = labels.get(type, [])
                target_labels.append(unit_labels)
        
        label_num = sum([len(items) for items in target_labels])
        if label_num == 0:
            return None

        # get used labels in batch data (sequence might be cutoff and remove some labels)
        target_labels_trim = []
        for batch_idx, mask in enumerate(ref_mask):
            ref_num = mask.sum()
            if ref_num == 0: continue
            assert len(target_labels[batch_idx]) >= ref_num
            target_labels_trim.extend(target_labels[batch_idx][:ref_num])
        target_labels_trim = [torch.tensor(label) for label in target_labels_trim]
        return target_labels_trim

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
                target_kpts = self.get_unit_labels(metas, ref_mask, 'keypoint')
                target_slices = self.get_label_slices(metas, ref_mask)
            except Exception as e:
                print(e)
                metas['type'] = 'pose'
                metas['ref_mask_filter'] = ref_mask
                save_wrong_data(f"wrong_ref_match", metas)
                raise ValueError('Error in get_unit_labels/seqs process, type = pose')
            
            pred_boxes1 = boxes_queries_logits1[ref_mask, :] # [b, n, 4]
            pred_boxes2 = boxes_queries_logits2[ref_mask, :] # [b, n, 4]
            pred_kpts = kpts_queries_logits[ref_mask, ...] # [b, n, num_body_points, 2]
            pred_kpts_cls = kpts_cls_logits[ref_mask, ...] # [b, n, num_body_points, 3]

            if ref_mask.sum() > 0 and target_boxes is not None:
                target_boxes = torch.stack(
                    target_boxes).to(pred_boxes1.device).to(pred_boxes1.dtype)
                target_kpts =  torch.stack(
                    target_kpts).to(pred_kpts.device).to(pred_kpts.dtype)

                box_weight_dict = {
                    "loss_bbox": self.config.box_config.bbox_loss_coefficient,
                    "loss_giou": self.config.box_config.giou_loss_coefficient}
                pose_weight_dict = {
                    "loss_ce": self.config.keypoint_config.cls_loss_coefficient,
                    "loss_keypoints": self.config.keypoint_config.keypoint_loss_coefficient,
                    "loss_oks": self.config.keypoint_config.oks_loss_coefficient
                }
                pose_weight_dict.update(box_weight_dict)

                loss_dict = self.criteria(pred_boxes2, target_boxes, pred_kpts, pred_kpts_cls, target_kpts, target_slices)
                loss = sum(loss_dict[k] * pose_weight_dict[k] for k in loss_dict.keys() if k in pose_weight_dict)

                if self.config.use_auxiliary_loss:
                    loss_dict_aux = self.box_decoder.criteria(pred_boxes1, target_boxes, target_slices)
                    loss_aux = sum(loss_dict_aux[k] * box_weight_dict[k] for k in loss_dict_aux.keys() if k in box_weight_dict)
                    loss += self.config.aux_loss_coefficient * loss_aux
            else:
                loss = pred_boxes1.sum() * 0.0 + pred_kpts.sum() * 0.0 + pred_boxes2.sum() * 0.0 + pred_kpts_cls.sum() * 0.0
        else:
            assert mode == 'predict'
            pred_kpts = []
            for queries_logits, mask in zip(kpts_queries_logits, ref_mask):
                keypoints = queries_logits[mask, :]
                pred_kpts.append(keypoints)

        return dict(
            loss=loss,
            preds=pred_kpts,
        )
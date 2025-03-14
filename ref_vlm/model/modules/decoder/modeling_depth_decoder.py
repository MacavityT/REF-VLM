# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .modeling_decoder import DecoderModel
from .configuration_depth_decoder import DepthDecoderConfig
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from transformers.models.detr import DetrConfig
from transformers.models.maskformer import MaskFormerConfig
from transformers.models.maskformer.modeling_maskformer import (
    DetrDecoder,
    MaskformerMLPPredictionHead,
    MaskFormerPixelDecoder,
    sigmoid_focal_loss,
    dice_loss,
    pair_wise_sigmoid_focal_loss,
    pair_wise_dice_loss
)

from transformers.utils import requires_backends, is_scipy_available

class DepthDecoderModel(DecoderModel):
    config_class = DepthDecoderConfig

    def __init__(self, config: DepthDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # pixel decoder
        self.pixel_decoder = MaskFormerPixelDecoder(
            in_features=config.encoder_input_dim[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=config.encoder_input_dim[:-1]
        )
        # visual features for query decoder cross attention
        self.queries_embedder = nn.Embedding(config.num_queries, config.d_model)
        self.in_proj_visual_feats = nn.Conv2d(
            config.encoder_input_dim[-1], 
            config.d_model, 
            kernel_size=1
        ) if config.encoder_input_dim[-1] != config.d_model else None

        # query decoder
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
        self.query_decoder = DetrDecoder(decoder_config)

        self.mask_embedder = MaskformerMLPPredictionHead(
            config.d_model, 
            config.d_model, 
            config.mask_feature_size
        )
        
        matcher = MaskDecoderGroupHungarianMatcher(
            cost_mask=config.mask_loss_coefficient,
            cost_dice=config.dice_loss_coefficient
        )
        if config.use_group_matcher:
            matcher = MaskDecoderGroupHungarianMatcher(
                cost_mask=config.mask_loss_coefficient,
                cost_dice=config.dice_loss_coefficient
            )
        else:
            matcher = None
        self.criteria = MaskDecoderLoss(matcher)
        self.post_init()

    def _expand_pixel_masks(self, metas, ref_mask):
        pixel_masks = torch.stack(metas['pixel_masks']) # [bs, h, w]

        expanded_masks = []
        for mask, ref in zip(pixel_masks, ref_mask):
            if ref.sum() == 0: continue
            h, w = mask.shape
            mask = mask.unsqueeze(0).expand(ref.sum(), h, w)
            expanded_masks.append(mask)
        if len(expanded_masks) == 0:
            return None
        expanded_masks = torch.cat(expanded_masks)
        return expanded_masks

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        tensors = []
        for input in inputs:
            if len(input.shape) == 2:
                input = input.unsqueeze(0)
            tensors.append(input)
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # compute final size
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks
    
    def forward(self, 
        visual_hidden_states,
        ref_hidden_states,
        visual_mask=None,
        ref_mask=None,
        metas=None,
        mode='loss'
    ):
        # remove other unit decode tasks
        decode_units = metas.get('decode_units', None)
        if decode_units is not None:
            batch_remove_mask = [unit != 'mask' for unit in decode_units]  
            # remove batch which task not mask deocde
            condition = torch.zeros_like(ref_mask).to(torch.bool)
            condition[batch_remove_mask, :] = True 
            ref_mask = ref_mask.masked_fill(condition, False)
        
        if mode != 'loss' and ref_mask.sum() == 0:
            return None

        # prepare visual hidden states
        visual_hidden_states = self.transform_visual_inputs(visual_hidden_states)
        if isinstance(visual_hidden_states, torch.Tensor):
             visual_hidden_states = self.blc2bchw(visual_hidden_states)
        else:
            visual_hidden_states = [self.blc2bchw(hidden_states) for hidden_states in visual_hidden_states]
        pixel_decoder_outputs = self.pixel_decoder(visual_hidden_states)
        image_features = visual_hidden_states[-1]
        pixel_embeddings = pixel_decoder_outputs.last_hidden_state

        if self.in_proj_visual_feats is not None:
            image_features = self.in_proj_visual_feats(image_features)
        
        visual_position_embedding, _ = self.visual_position_encoding(image_features)
        batch_size, num_channels, height, width = image_features.shape
        # rearrange both image_features and object_queries "b c h w -> b (h w) c"
        image_features = image_features.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        
        # prepare learnable queries
        batch_size = ref_hidden_states.shape[0]
        ref_hidden_states = self.in_proj_queries(ref_hidden_states)
        ref_hidden_states, ref_mask = self.padding_ref_inputs(
            ref_hidden_states,
            ref_mask
        )
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        assert ref_hidden_states.shape == query_position_embeddings.shape

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.query_decoder(
            inputs_embeds=ref_hidden_states,
            attention_mask=None, # could be ref_mask, still with some bugs
            object_queries=visual_position_embedding,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None, # mask for pixel values
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs[0]
        mask_embeddings = self.mask_embedder(sequence_output)
        masks_queries_logits = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings)
        
        loss = None
        if mode == 'loss':   
            try:
                pred_masks = masks_queries_logits[ref_mask, :]
                target_masks = self.get_unit_labels(metas, ref_mask, 'mask')
                target_slices = self.get_label_slices(metas, ref_mask)
                pixel_masks = self._expand_pixel_masks(metas, ref_mask)
            except Exception as e:
                from ref_vlm.model.utils import save_wrong_data
                print(e)
                metas['type'] = 'mask'
                metas['ref_mask_filter'] = ref_mask 
                save_wrong_data(f"wrong_ref_match", metas)
                raise ValueError('Error in get_unit_labels/seqs process, type = mask')

            if ref_mask.sum() > 0 and target_masks is not None:
                target_masks, _ = self._pad_images_to_max_in_batch(target_masks)
                target_masks = target_masks.to(pred_masks.device).to(pred_masks.dtype)
                loss_dict = self.criteria(pred_masks, target_masks, target_slices, pixel_masks)
                weight_dict = {
                    "loss_mask": self.config.mask_loss_coefficient,
                    "loss_dice": self.config.dice_loss_coefficient}
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            else:
                loss = pred_masks.sum() * 0.0
        else:
            masks_queries_logits = torch.sigmoid(masks_queries_logits)
            pred_masks = []
            for queries_logits, mask in zip(masks_queries_logits, ref_mask):
                masks = queries_logits[mask, :]
                pred_masks.append(masks)

        return dict(
            loss=loss,
            preds=pred_masks
        )
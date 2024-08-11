# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from torch import Tensor
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .modeling_decoder import DecoderModel
from .configuration_mask2former_decoder import MaskDecoderConfig
from typing import Dict, List, Optional, Tuple

from transformers.models.mask2former import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerSinePositionEmbedding,
    Mask2FormerModel,
    Mask2FormerPixelLevelModule, #not used, wrapper of Mask2FormerPixelDecoder
    Mask2FormerPixelDecoder,
    Mask2FormerTransformerModule, # not used, wrapper of Mask2FormerMaskedAttentionDecoder
    Mask2FormerMaskedAttentionDecoder,
    sigmoid_cross_entropy_loss,
    dice_loss,
    sample_point
)

class MaskQueryDecoder(nn.Module):
    """
    The Mask2Former-like transformer module.
    """

    def __init__(self, in_features: int, config: MaskDecoderConfig):
        super().__init__()
        hidden_dim = config.d_model
        self.num_feature_levels = 3
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.d_model, hidden_dim)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())
        
        decoder_config = Mask2FormerConfig(
            mask_feature_size = config.mask_feature_size,
            dropout = config.dropout,
            decoder_layers = config.decoder_layers,
            hidden_dim=config.d_model,
            num_attention_heads=config.decoder_attention_heads,
            pre_norm=config.pre_norm,
            activation_function=config.activation_function,
            dim_feedforward=config.decoder_ffn_dim
        )
        self.decoder = Mask2FormerMaskedAttentionDecoder(decoder_config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        query_features: Tensor,
    ):
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # Flatten (batch_size, num_channels, height, width) -> (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # [num_queries, batch_size, num_channels]
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = query_features.permute(1, 0, 2)
        assert query_features.shape == query_embeddings.shape

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        return decoder_output

class MaskDecoderModel(DecoderModel):
    config_class = MaskDecoderConfig

    def __init__(self, config: MaskDecoderConfig):
        super().__init__(config)
        self.config = config

        # visual feats
        pixel_decoder_config = Mask2FormerConfig(
            feature_size=config.feature_size,
            mask_feature_size=config.mask_feature_size,
            feature_strides=config.feature_strides,
            common_stride=config.common_stride,
            encoder_feedforward_dim=config.encoder_feedforward_dim,
            dropout=config.dropout,
            num_attention_heads=config.decoder_attention_heads,
            encoder_layers=config.encoder_layers,
        )
        self.pixel_decoder = Mask2FormerPixelDecoder(
            pixel_decoder_config, 
            feature_channels=self.in_channels
        )

        # query feats
        self.in_proj_queries = nn.Linear(config.quries_input_dim, config.d_model)
        self.query_decoder = MaskQueryDecoder(
            in_features=config.feature_size,
            config=config
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def transform_visual_inputs(self, inputs):
        visual_hidden_states = super().transform_visual_inputs(inputs)
        
        _visual_hidden_states = []
        for hidden_states in visual_hidden_states:
            b, l, c = hidden_states.shape
            grid_size = int(math.sqrt(l))
            hidden_states = hidden_states.view(b, grid_size, grid_size, c)
            hidden_states = hidden_states.permute(0, 3, 1, 2) # b, c, h, w
            _visual_hidden_states.append(
                hidden_states
            )
        return _visual_hidden_states

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # Adapted from nested_tensor_from_tensor_list() in original implementation
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
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

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates

    def compute_loss_mask(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)

        # No need to upsample predictions as we are using normalized coordinates
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        # Sample point coordinates
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
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
        pixel_level_module_output = self.pixel_decoder(
            visual_hidden_states, 
            output_hidden_states=False
        )

        # prepare learnable queries
        ref_hidden_states = self.in_proj_queries(ref_hidden_states)
        transformer_module_output = self.query_decoder(
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state
        )
        pred_masks = transformer_module_output.masks_queries_logits[-1]

        loss = None
        if mode == 'loss':
            pred_masks = pred_masks[ref_mask, :]
            target_masks = self.get_unit_labels(metas, ref_mask, 'box')
            
            if ref_mask.sum() > 0 and target_masks is not None:
                target_masks = target_masks.to(pred_masks.device).to(pred_masks.dtype)
                loss_dict = self.compute_loss_mask(pred_masks, target_masks)
                weight_dict = {
                    "loss_mask": self.config.mask_loss_coefficient,
                    "loss_dice": self.config.dice_loss_coefficient}
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            else:
                loss = torch.tensor(0).to(visual_hidden_states.device).to(visual_hidden_states.dtype)
        else:
            pred_masks = pred_masks[ref_mask, :]
            pred_masks = torch.sigmoid(pred_masks)

        return dict(
            loss=loss,
            preds=pred_masks
        )
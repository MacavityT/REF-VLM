import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from .build_sam import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
from .utils.transforms import ResizeLongestSide
from ref_vlm.model.modules import MaskDecoderConfig
from ref_vlm.model.modules.decoder import DecoderModel
from ref_vlm.model.modules.decoder.modeling_mask_decoder import MaskDecoderLoss, MaskDecoderGroupHungarianMatcher

class SamPreprocessor(ResizeLongestSide):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    def __init__(self, target_length=1024):
        super().__init__(target_length=target_length)
        self.img_size = target_length

    def preprocess(self, image, *args, **kwargs):
        """Normalize pixel values and pad to a square input."""
        image = np.asarray(image)
        image = self.apply_image(image)
        x = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        # Normalize colors
        x = (x - self.IMG_MEAN) / self.IMG_STD

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return dict(
            pixel_values = [x]
        )


class SamPlug(DecoderModel):
    config_class = MaskDecoderConfig

    SAM_BUILDER = {
        'base': build_sam_vit_b,
        'large': build_sam_vit_l,
        'huge': build_sam_vit_h
    }

    def __init__(self, config: MaskDecoderConfig, version='huge', checkpoint=None, freeze_mask_decoder=False):
        super().__init__(config)
        self.sam = self.SAM_BUILDER[version](checkpoint)
        self.config = config

        self.initialize_sam(freeze_mask_decoder)
        # ref token input projection
        in_dim, out_dim = self.config.quries_input_dim, self.config.d_model
        project_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.in_proj_queries = nn.Sequential(*project_layers)

        matcher = MaskDecoderGroupHungarianMatcher(
            cost_mask=self.config.mask_loss_coefficient,
            cost_dice=self.config.dice_loss_coefficient
        )
        if self.config.use_group_matcher:
            matcher = MaskDecoderGroupHungarianMatcher(
                cost_mask=self.config.mask_loss_coefficient,
                cost_dice=self.config.dice_loss_coefficient
            )
        else:
            matcher = None
        self.criteria = MaskDecoderLoss(matcher)

    def initialize_sam(self, freeze_mask_decoder):
        # Freezing visual model parameters
        for param in self.sam.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if not freeze_mask_decoder:
            self.sam.mask_decoder.train()
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = True

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
        assert 'pixel_values_tower' in metas.keys()
        image_features = self.sam.image_encoder(metas['pixel_values_tower'])

        # prepare learnable queries
        ref_hidden_states = self.in_proj_queries(ref_hidden_states)
        ref_hidden_states, ref_mask = self.padding_ref_inputs(
            ref_hidden_states,
            ref_mask
        )

        # get predict masks
        pred_masks_logits = []
        for ref_tokens, valid_mask, image_embedding in zip(
            ref_hidden_states, ref_mask, image_features):
            ref_tokens = ref_tokens[valid_mask]
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=ref_tokens.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(ref_hidden_states.dtype)
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=image_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False)
            pred_masks_logits.append(low_res_masks[:, 0, ...])
        pred_masks_logits = torch.cat(pred_masks_logits, dim=0)
        
        loss = None
        if mode == 'loss':   
            try:
                pred_masks = pred_masks_logits
                target_masks = self.get_unit_labels(metas, ref_mask, 'mask')
                target_slices = self.get_label_slices(metas, ref_mask)
                pixel_masks = self._expand_pixel_masks(metas, ref_mask)
            except Exception as e:
                from ref_vlm.model.utils import save_wrong_data
                print(e)
                metas['type'] = 'mask'
                metas['ref_mask_filter'] = ref_mask 
                save_wrong_data(f"wrong_ref_match", metas)
                raise ValueError('Error in get_unit_labels/seqs process, type = sam mask')

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
            pred_masks_logits = torch.sigmoid(pred_masks_logits)

            start = 0
            pred_masks = []
            for mask in ref_mask:
                end = start + mask.sum().cpu().item()
                sigmoid_mask = pred_masks_logits[start:end]
                pred_masks.append(sigmoid_mask)
                start = end
            assert end == pred_masks_logits.shape[0]

        return dict(
            loss=loss,
            preds=pred_masks
        )

def build_sam_preprocessor(target_length=1024):
    return SamPreprocessor(target_length)

def build_sam_plug(version='huge', checkpoint=None, freeze_mask_decoder=False, **kwargs):
    config = MaskDecoderConfig(**kwargs)
    return SamPlug(config, version, checkpoint, freeze_mask_decoder)
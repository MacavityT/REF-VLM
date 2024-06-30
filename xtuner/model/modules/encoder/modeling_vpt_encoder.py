# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_vpt_encoder import VPTEncoderConfig

class VPTEncoderModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = VPTEncoderConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: VPTEncoderConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.config = config
        self.patch_size = int(math.sqrt(config.mask_patch_len))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.visual_hidden_size,
            out_channels=config.visual_hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = config.vis_feats_len // config.mask_patch_len
        self.position_embedding = nn.Embedding(self.num_patches, config.visual_hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1, -1)), persistent=False)

        # # projector
        # modules = [
        #     nn.Linear(
        #         config.visual_hidden_size,
        #         config.llm_hidden_size,
        #         bias=config.bias)
        # ]
        # for _ in range(1, config.depth):
        #     modules.append(ACT2FN[config.hidden_act])
        #     modules.append(
        #         nn.Linear(
        #             config.llm_hidden_size,
        #             config.llm_hidden_size,
        #             bias=config.bias))
        # self.model = nn.Sequential(*modules)


    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.position_embedding.register_forward_hook(make_inputs_require_grad)
        self.patch_embedding.register_forward_hook(make_inputs_require_grad)
        # self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, VPTEncoderModel):
            module.gradient_checkpointing = value

    def pad_regions(self, regions, dtype, device):
        region_count = []
        max_h, max_w = 0, 0
        for regions_in_batch in regions:
            if regions_in_batch is None:
                region_count.append(0)
                continue
            assert isinstance(regions_in_batch, list)
            region_count.append(len(regions_in_batch))
            for region in regions_in_batch:
                h, w = region.shape
                if h > max_h: max_h = h
                if w > max_w: max_w = w

        batch_size = len(regions)
        max_num = max(region_count)
        tensor_regions = torch.zeros((batch_size, max_num, max_h, max_w), dtype=dtype)
        for batch_idx, num in enumerate(region_count):
            if num == 0: continue
            regions_in_batch = regions[batch_idx]
            for region_index, region in enumerate(regions_in_batch):
                if isinstance(region, np.ndarray):
                    region = torch.from_numpy(region).to(dtype)
                elif isinstance(region, torch.Tensor):
                    region = region.to(dtype)
                else:
                    raise ValueError("VPT region type error!")
                tensor_regions[batch_idx, region_index, ...] = region
        tensor_regions = tensor_regions.to(device)
        return tensor_regions, region_count


    def mask_patch_feats(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask = (mask > 0).to(mask.dtype).to(mask.device)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8
        mask = mask / denorm

        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        
        if self.config.strategy == 'pooling':
            div = math.sqrt(self.num_patches)
            x_patches = x.reshape(b, c, self.num_patches, h // div, w // div)
            mask_patches = mask.reshape(b, q, self.num_patches, h // div, w // div)
            b, c, n, h, w = x_patches.shape
            b, q, n, h, w = mask_patches.shape
            vpt_feats = torch.einsum(
                "bcnhw,bqnhw->bqnc",
                x,
                mask_patches,
            )
        elif self.config.strategy == 'embedding':
            target_dtype = self.patch_embedding.weight.dtype
            x = x.unsqueeze(1) # [b, 1, c, h, w]
            mask = mask.unsqueeze(2) # [b, q, 1, h, w]
            vpt_feats = x * mask # [b, q, c, h, w]
            vpt_feats = vpt_feats.view(-1, c, h, w) # [b*q, c, h, w]
            vpt_feats = self.patch_embedding(vpt_feats.to(dtype=target_dtype)) # shape = [b*q, c, 3, 3]
            vpt_feats = vpt_feats.flatten(2).transpose(1, 2) # [b*q, c, n]->[b*q, n, c],  n=9
            vpt_feats = vpt_feats.view(b, q, self.num_patches, c)
        else:
            raise NotImplementedError
        return vpt_feats

    # def project(self, x):
    #     if self.gradient_checkpointing and self.training:
    #         layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
    #     else:
    #         layer_outputs = self.model(x)
    #     return layer_outputs

    def forward(self, x, regions, return_dict=True):
        """
        To extract the region feartures based on the region mask.
        Args:
            x(`tensor`): [B, L, C], image feature -> [batch_size, 256, 1024]
            regions(`List[List[torch.Tensor]]`): mask
        Returns:
            region features: [B, Q, N, C]
            return the mask patch pooling features based on the region mask.
        """
        b, l, c = x.shape
        w = h = int(math.sqrt(x.shape[1]))
        assert x.size(0) == len(regions)

        # pad regions list to tensor
        regions, vpt_count = self.pad_regions(regions, x.dtype, x.device) # b, q, h, w
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b, c, h, w
        vpt_feats = self.mask_patch_feats(x, regions)  # b, q, n, c
        vpt_feats = vpt_feats + self.position_embedding(self.position_ids)
        # vpt_feats = self.project(vpt_feats)

        if return_dict:
            result = dict(
                vpt_feats = vpt_feats,
                vpt_count = vpt_count
            )
        else:
            result = (vpt_feats, vpt_count)
        return result
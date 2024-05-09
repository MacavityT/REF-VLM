# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VPTProcessor:
    def __init__(self, vpt_grid=(3, 3)):
        """
        Args:
            vpt_grid: shape of visual prompts grid
        """
        self.vpt_grid = vpt_grid

    def pad_regions(self, regions, dtype):
        region_count = []
        max_h, max_w = 0, 0
        for regions_in_batch in regions:
            if regions_in_batch is None:
                region_count.append(0)
                continue
            assert isinstance(regions_in_batch, list)
            region_count.append(len(regions_in_batch))
            for region in regions_in_batch:
                assert isinstance(region, torch.Tensor)
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
                tensor_regions[batch_idx, region_index, ...] = region.to(dtype)
            
        return tensor_regions, region_count

    def mask_patch_pooling(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask = (mask > 0).to(mask.dtype)
        denorm = mask_patches.sum(dim=(-1, -2), keepdim=True) + 1e-8
        normed_mask = mask / denorm

        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        grid_h, grid_w = self.vpt_grid
        n = grid_h * grid_w

        x_patches = x.reshape(b, c, n, h // grid_h, w // grid_w)
        mask_patches = normed_mask.reshape(b, q, n, h // grid_h, w // grid_w)

        b, c, n, h, w = x_patches.shape
        b, q, n, h, w = mask_patches.shape
        mask_pooled_x = torch.einsum(
            "bcnhw,bqnhw->bqnc",
            x,
            mask_patches,
        )
        return mask_pooled_x

    def __call__(self, x, regions, return_dict=True):
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
        regions, vpt_count = self.pad_regions(regions, x.dtype) # b, q, h, w
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b, c, h, w
        vpt_feats = self.mask_patch_pooling(x, regions)  # b, q, n, c

        if return_dict:
            result = dict(
                vpt_feats = vpt_feats,
                vpt_count = vpt_count
            )
        else:
            result = (vpt_feats, vpt_count)
        return result




# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from xtuner.utils.constants import DEFAULT_PAD_TOKEN_INDEX
    
# def transform_bbox_2_mask(self, bboxes, image_size, device, data_type=torch.float):
#     batch_masks = []
#     for bbox in bboxes:
#         mask = torch.zeros((image_size, image_size), dtype=data_type).to(device)
#         x1, y1, x2, y2 = bbox
#         # x2, y2 = int(x1 + w), int(y1 + h)
#         mask[int(x1):int(x2),int(y1):int(y2)] = 1
#         batch_masks.append(mask)
#     return torch.stack(batch_masks, dim=0)

class VPTProcessor:
    def __init__(self, in_dim=1024, out_dim=4096, patch_size=14, image_size=224, vpt_div=9):
        """
        Args:
            image_size: the image size of ViT when inputing images
            patch_size: the patch size of ViT when encoding images
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.vpt_div = vpt_div

    def get_region_features(self, pad_val):
        return 0

    def mask_patch_pooling(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        mask = (mask > 0).to(mask.dtype)
        denorm = mask_patches.sum(dim=(-1, -2), keepdim=True) + 1e-8
        normed_mask = mask / denorm

        div = self.vpt_div
        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        n = div**2

        x_patches = x.reshape(b, c, n, h//div, w//div)
        mask_patches = normed_mask.reshape(b, q, n, h//div, w//div)

        b, c, n, h, w = x_patches.shape
        b, q, n, h, w = mask_patches.shape



        # import torch

        # # 假设你有一个普通的矩阵
        # M = torch.tensor([[0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 2.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 3.0]])

        # # 提取非零元素的索引和值
        # indices = torch.nonzero(M)
        # values = M[indices[:, 0], indices[:, 1]]

        # # 使用提取的索引和值创建稀疏张量
        # M_sparse = torch.sparse_coo_tensor(indices.t(), values, M.size())

        # print(M_sparse)





        mask_pooled_x = torch.einsum(
            "bcnhw,bqnhw->bqnc",
            x,
            mask_patches,
        )
        return mask_pooled_x

    def __call__(self, x, regions):
        """
        To extract the region feartures based on the region mask.
        Args:
            x(`tensor`): [B, L, C], image feature -> [batch_size, 256, 1024]
            regions(`List[List[tensor]]`): [B, Q, H, W], mask
        Returns:
            region features: [B, Q, N, C]
            return the mask patch pooling features based on the region mask.
        """
        b, l, c = x.shape
        w = h = int(math.sqrt(x.shape[1]))
        assert x.size(0) == len(regions)

        # conver regions list to tensor
        region_feats = self.get_region_features(regions) # b, q, h, w


        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b, c, h, w
        region_feats = self.mask_patch_pooling(x, region_feats)  # b, q, n, c
        return region_feats




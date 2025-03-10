import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import ast

from .configuration_unipose import UniPoseConfig
from .modeling_unipose import UniPose
from ref_vlm.model.modules import MaskDecoderConfig
from ref_vlm.model.modules.decoder import DecoderModel
from ref_vlm.model.modules.decoder.modeling_pose_decoder import PoseDecoderLoss
from ref_vlm.model.modules.decoder.modeling_box_decoder import BoxDecoderGroupHungarianMatcher


class UniposePlug(DecoderModel):
    config_class = UniPoseConfig

    def __init__(self, unipose, config):
        super().__init__()
        self.unipose = unipose
        self.config = config
        self.unipose = UniPose(config)
        self.init_unipose(config.checkpoint)

        matcher = BoxDecoderGroupHungarianMatcher(
            bbox_cost=self.config.bbox_loss_coefficient,
            giou_cost=self.config.giou_loss_coefficient
        )
        self.criteria = PoseDecoderLoss(matcher)

    def init_unipose(self, checkpoint):
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            self.unipose.load_state_dict(state_dict, strict=False)

    def forward(self, 
        visual_hidden_states,
        ref_hidden_states,
        visual_mask=None,
        ref_mask=None,
        metas=None,
        mode='loss'
    ):
        
        num_patches = emb_select.sum(-1) // self.num_embs 
        # this is for obj class patches
        max_num_obj_patches = 100
        # this is for pose class patches
        max_num_kpt_patches = 100

        # [bs, max_num_obj/kpt_patches, num_embs, c], [bs, max_num_obj/kpt_patches]
        obj_querys = torch.zeros((batch_size, max_num_obj_patches, self.num_embs, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        obj_query_masks = torch.zeros((batch_size, max_num_obj_patches), dtype=torch.bool, device=hidden_states.device)
        kpt_querys = torch.zeros((batch_size, max_num_kpt_patches, self.num_embs, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        kpt_query_masks = torch.zeros((batch_size, max_num_kpt_patches), dtype=torch.bool, device=hidden_states.device)
        for batch_idx in range(batch_size):
            num_objcls = len(img_metas[batch_idx]['id2index'])
            num_kpts = num_patches[batch_idx] - num_objcls
            if num_objcls != 0 and num_kpts != 0:
                text_query_i = hidden_states[batch_idx, emb_select[batch_idx], :].reshape(-1, self.num_embs, hidden_size)
                obj_querys[batch_idx, :num_objcls] = text_query_i[:num_objcls, ...] 
                obj_query_masks[batch_idx, :num_objcls] = 1                
                kpt_querys[batch_idx, :num_kpts] = text_query_i[num_objcls:, ...]  
                kpt_query_masks[batch_idx, :num_kpts] = 1               




def build_unipose_plug(unipose_cfg, checkpoint, **kwargs):
    def read_params(file_path):
        params = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去掉首尾空白字符
                if line and not line.startswith("#"):  # 忽略空行和注释行
                    try:
                        key, value = map(str.strip, line.split('=', 1))  # 按等号分割
                        # 使用 ast.literal_eval 转换值为合适的 Python 数据类型
                        try:
                            value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            value = value.strip('"').strip("'")  # 保留为字符串
                        params[key] = value
                    except ValueError:
                        print(f"无法解析参数行: {line}")
        return params

    parameters = read_params(unipose_cfg)
    parameters.update(kwargs)

    config = UniPoseConfig(**parameters)
    return UniposePlug(config)
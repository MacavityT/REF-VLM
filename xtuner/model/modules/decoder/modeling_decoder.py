import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple

class DecoderPositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(x.shape) == 3:
            b, l, c = x.shape
            grid_size = int(math.sqrt(l))
            x = x.view(b, grid_size, grid_size, c)
            x = x.permute(0, 3, 1, 2) # b, c, h, w
            b, c, h, w = x.shape
        elif len(x.shape) == 4:
            b, c, h, w = x.shape
        else:
            raise ValueError("input hidden states with wrong shape.")  

        if mask is None:
            mask = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
            mask = (mask > 0).to(torch.bool).to(x.device)

        y_embed = mask.cumsum(1)
        x_embed = mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # flatten
        pos = pos.permute(0, 2, 3, 1).view(b, -1, c).to(x.dtype) # b, target_length, c
        mask = mask.flatten(1)
        return pos, mask

class DecoderModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    base_model_prefix = 'model'
    supports_gradient_checkpointing = False

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self._init_inputs(
            config.encoder_input_dim,
            config.encoder_input_index,
            config.encoder_input_transform
        )
        self.visual_position_encoding = DecoderPositionEmbedding(
            num_pos_feats=config.d_model // 2, 
            normalize=True
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PreTrainedModel):
            module.gradient_checkpointing = value

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def blc2bchw(self, x):
        if len(x.shape) == 3:
            b, l, c = x.shape
            grid_size = int(math.sqrt(l))
            x = x.view(b, grid_size, grid_size, c)
            x = x.permute(0, 3, 1, 2) # b, c, h, w
        return x
    
    def bchw2blc(self, x):
        if len(x.shape) == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).view(b, -1, c) # b, L, c
        return x

    def resize(self, x, target_size):
        x = F.interpolate(
            x, 
            size=target_size,
            mode="bilinear",
            align_corners=False
        )
        return x

    def transform_visual_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [self.blc2bchw(x) for x in inputs]
            target_size = inputs[0].shape[2:]
            upsampled_inputs = [
                self.resize(
                    x,
                    target_size=target_size
                ) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def padding_ref_inputs(self, hidden_states, hidden_states_mask):
        batch, length, dim = hidden_states.shape

        if length > self.config.num_queries:
            padded_hidden_states = hidden_states[:, :self.config.num_queries, :]
            padded_mask = hidden_states_mask[:, :self.config.num_queries]
        else:
            padded_hidden_states = torch.zeros(
                (batch, self.config.num_queries, dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            padded_mask = torch.zeros(
                (batch, self.config.num_queries),
                device=hidden_states.device,
                dtype=torch.bool
            )
            padded_hidden_states[:, :length, :] = hidden_states
            padded_mask[:, :length] = hidden_states_mask

        return padded_hidden_states, padded_mask

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
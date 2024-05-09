# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .vpt_processor import VPTProcessor
from .configuration_projector import ProjectorConfig

# just placeholder

# class CLIPVisionEmbeddings(nn.Module):
#     def __init__(self, config: CLIPVisionConfig):
#         super().__init__()
#         self.config = config
#         self.embed_dim = config.hidden_size
#         self.image_size = config.image_size
#         self.patch_size = config.patch_size

#         self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

#         self.patch_embedding = nn.Conv2d(
#             in_channels=config.num_channels,
#             out_channels=self.embed_dim,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             bias=False,
#         )

#         self.num_patches = (self.image_size // self.patch_size) ** 2
#         self.num_positions = self.num_patches + 1
#         self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
#         self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

#     def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
#         batch_size = pixel_values.shape[0]
#         target_dtype = self.patch_embedding.weight.dtype
#         patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
#         patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

#         class_embeds = self.class_embedding.expand(batch_size, 1, -1)
#         embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
#         embeddings = embeddings + self.position_embedding(self.position_ids)
#         return embeddings

class ProjectorModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = ProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        modules = [
            nn.Linear(
                config.visual_hidden_size,
                config.llm_hidden_size,
                bias=config.bias)
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.model = nn.Sequential(*modules)

        # visual prompt encoding
        if 'vpt_config' in config:
            self.vpt_processor = VPTProcessor(**config.vpt_config)
            self.num_vpt_patches = config.vpt_config['vpt_grid'][0] * config.vpt_config['vpt_grid'][1]
            self.vpt_position_embedding = nn.Embedding(self.num_vpt_patches, self.visual_hidden_size)
            self.register_buffer("vpt_position_ids", torch.arange(self.num_vpt_patches).expand((1, -1)), persistent=False)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ProjectorModel):
            module.gradient_checkpointing = value

    def visual_forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs

    def forward_vpt(self, x):
        embeddings = x + self.vpt_position_embedding(self.vpt_position_ids)
        layer_outputs = self.forward(embeddings)
        return layer_outputs

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
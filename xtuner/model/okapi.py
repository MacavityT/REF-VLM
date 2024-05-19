# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict
import json
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.logging import print_log
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig,GenerationConfig, StoppingCriteriaList

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from .modules import (
    ProjectorConfig, ProjectorModel,
    VPTEncoderConfig, VPTEncoderModel,
    dispatch_modules
    )
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad, traverse_dict,
                    prepare_inputs_labels_for_multimodal)
from xtuner.utils.constants import SPECIAL_TOKENS

class OkapiModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 visual_decoder=None,
                 vpt_encoder=None,
                 projector=None,
                 tokenizer=None,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_projector=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 cutoff_len=None,
                 max_position_embeddings=None):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_projector = freeze_projector
        self.cutoff_len = cutoff_len
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)
            self.tokenizer = self._prepare_tokenizer(tokenizer)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            # generate config
            default_generation_kwargs = dict(
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.eos_token_id)
            self.max_new_tokens = 512
            self.gen_config = GenerationConfig(**default_generation_kwargs)
            self.stop_criteria = StoppingCriteriaList()

            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
            #taiyan TODO: 增加 visual decoder 初始化
            # self.visual_decoder = 
            if vpt_encoder is not None:
                self.vpt_encoder = self._build_from_cfg_or_module(
                    vpt_encoder)
            if projector is not None:
                self.projector = self._build_from_cfg_or_module(
                    projector)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if projector is None:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth
            )
            self.projector = ProjectorModel(projector_config).to(
                self.visual_encoder.dtype)
        if vpt_encoder is None:
            image_size = self.visual_encoder.config.image_size
            patch_size = self.visual_encoder.config.patch_size
            vis_feats_len = (image_size // patch_size) ** 2
            num_patches = 9
            vpt_encoder_config = VPTEncoderConfig(
                strategy='embedding',
                vis_feats_len=vis_feats_len,
                mask_patch_len=vis_feats_len // num_patches,
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth
            )
            self.vpt_encoder = VPTEncoderModel(vpt_encoder_config).to(
                self.visual_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        if self.freeze_projector:
            self.projector.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()
            self.vpt_encoder.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

    @staticmethod
    def _prepare_tokenizer(tokenizer_cfg):
        tokenizer = BUILDER.build(tokenizer_cfg)
        tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        return tokenizer

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.vpt_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.vpt_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        # Step 4. VPT Encoder
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'vpt_encoder.' in k})
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config',
                             'Starcoder2Config', 'Starcoder2Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Starcoder2Config', 'Starcoder2Config')

        if SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch.bfloat16 \
                if torch.cuda.is_bf16_supported() else torch.float16
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):

        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype),
                output_hidden_states=True)
            selected_feats = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]

            if 'visual_prompts' in data:
                visual_prompts = self.vpt_encoder(
                    selected_feats,
                    regions = data['visual_prompts'], 
                    return_dict = True
                )
                data.update(visual_prompts)

            pixel_values = self.projector(selected_feats)
            data['pixel_values'] = pixel_values

            if mode == 'predict':
                labels_mask = (data['labels'].detach().cpu().numpy()[0] == IGNORE_INDEX).tolist()
                data['input_ids'] = data['input_ids'][0][:labels_mask.index(False)].unsqueeze(0)
                data['labels'] = None
                data['attention_mask'] = None
                data['position_ids'] = None
            
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
            if self.cutoff_len is not None:
                if data['inputs_embeds'].shape[1] > self.cutoff_len:
                    data['inputs_embeds'] = data['inputs_embeds'][:, :self.cutoff_len, :]
                    data['labels'] = data['labels'][:, :self.cutoff_len]
                    data['position_ids'] = data['position_ids'][:, :self.cutoff_len]
                    data['attention_mask'] = data['attention_mask'][:, :self.cutoff_len]

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs
    
    # TODO： Aaron add
    def predict(self, data, data_samples=None):

        generate_ids = self.llm.generate(
                **data,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria)

        generate_ids_dict = [{'generate_ids':generate_id} for generate_id in generate_ids]
        return generate_ids_dict

    # def predict(self, data, data_samples=None):
        
    #     outputs = self.llm(**data)
    #     logits_dict = [{'logits': logits} for logits in outputs.logits]
    #     return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)

        #taiyan TODO: add decoder loss 部分
        hidden_states = outputs.hidden_states

        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

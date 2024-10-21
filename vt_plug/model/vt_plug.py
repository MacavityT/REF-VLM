# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.logging import print_log
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, GenerationConfig, StoppingCriteriaList

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2, dispatch_modules
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad, traverse_dict)

from .modules import *
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad, traverse_dict,
                    prepare_inputs_labels_for_multimodal,
                    save_wrong_data)

from ..utils.constants import (
    SPECIAL_TOKENS,
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    VISUAL_REFERENCE_TOKEN,
)

DECODER_CONFIG_CLASS = {
    'box': BoxDecoderConfig,
    'mask': MaskDecoderConfig,
    'pose': PoseDecoderConfig,
    'depth': DepthDecoderConfig
}

DECODER_MODEL_CLASS = {
    'box': BoxDecoderModel,
    'mask': MaskDecoderModel,
    'pose': PoseDecoderModel,
    'depth': DepthDecoderModel
}

TOKEN_MASK_IDS = {
    'ref_masks': 1,
    'bou_masks': 2,
    'eou_masks': 3,
    'bop_masks': 4,
    'eop_masks': 5,
}

class VTPlugModel(BaseModel):

    def __init__(self,
                 llm,
                 tokenizer=None,
                 visual_encoder=None,
                 visual_tower=None,
                 vpt_encoder=None,
                 projector=None,
                 ref_adapter=None,
                 visual_decoder=None,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_projector=False,
                 freeze_vpt_encoder=False,
                 freeze_ref_adapter=False,
                 freeze_visual_decoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 cutoff_len=None,
                 max_position_embeddings=None,
                 loss_coefficient=None):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_projector = freeze_projector
        self.freeze_vpt_encoder = freeze_vpt_encoder
        self.freeze_ref_adapter = freeze_ref_adapter
        self.freeze_visual_decoder = freeze_visual_decoder
        self.cutoff_len = cutoff_len
        self.loss_coefficient = loss_coefficient
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)
            self.tokenizer = self._prepare_tokenizer(tokenizer)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            # token labels
            tokens_in_labels = [
                BOT_TOKEN, EOT_TOKEN, 
                BOU_TOKEN, EOU_TOKEN, 
                BOV_TOKEN, EOV_TOKEN,
                VISUAL_REFERENCE_TOKEN,
                PHRASE_ST_PLACEHOLDER_STAGE2,
                PHRASE_ED_PLACEHOLDER_STAGE2
            ]
            self.token_ids = {}
            for token in tokens_in_labels:
                self.token_ids[token] = self.tokenizer.convert_tokens_to_ids(token)
            # generate config
            default_generation_kwargs = dict(
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                output_hidden_states=True,
                return_dict_in_generate=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.eos_token_id)
            self.max_new_tokens = 512
            self.gen_config = GenerationConfig(**default_generation_kwargs)
            self.stop_criteria = StoppingCriteriaList()

            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder).to(self.llm.dtype)
            if visual_tower is not None:
                self.visual_tower = self._build_from_cfg_or_module(
                    visual_tower).to(self.llm.dtype)
            else:
                self.visual_tower = None
            if projector is not None:
                self.projector = self._build_from_cfg_or_module(
                    projector).to(self.llm.dtype)
            else:
                self.projector = None
            if vpt_encoder is not None and 'type' in vpt_encoder:
                self.vpt_encoder = self._build_from_cfg_or_module(
                    vpt_encoder).to(self.llm.dtype)
            else:
                self.vpt_encoder = None
            if ref_adapter is not None and 'type' in ref_adapter:
                self.ref_adapter = self._build_from_cfg_or_module(
                    ref_adapter).to(self.llm.dtype)
            else:
                self.ref_adapter = None

            self.visual_decoder = nn.ModuleDict()
            if visual_decoder is not None:
                assert isinstance(visual_decoder, dict)
                for decoder_type, decoder_config in visual_decoder.items():
                    if 'type' in decoder_config:
                        self.visual_decoder[decoder_type] = \
                            self._build_from_cfg_or_module(decoder_config).to(self.llm.dtype)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if self.projector is None:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth
            )
            self.projector = ProjectorModel(projector_config).to(
                self.llm.dtype)
        if vpt_encoder is not None and \
            'type' not in vpt_encoder:
            vpt_encoder_config = VPTEncoderConfig(**vpt_encoder)
            self.vpt_encoder = VPTEncoderModel(vpt_encoder_config).to(
                self.llm.dtype)
        if ref_adapter is not None and \
            'type' not in ref_adapter:
            ref_adapter_config = REFAdapterConfig(**ref_adapter)
            self.ref_adapter = REFAdapterModel(ref_adapter_config).to(
                self.llm.dtype)
        if visual_decoder is not None:
            assert isinstance(visual_decoder, dict)
            for decoder_type, decoder_config in visual_decoder.items():
                if 'type' not in decoder_config:
                    assert decoder_type not in self.visual_decoder.keys()
                    decoder_config = DECODER_CONFIG_CLASS[decoder_type](**decoder_config)
                    self.visual_decoder[decoder_type] = \
                        DECODER_MODEL_CLASS[decoder_type](decoder_config).to(
                            self.llm.dtype)
        if len(self.visual_decoder) == 0:
            self.visual_decoder = None

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        if self.freeze_projector:
            self.projector.requires_grad_(False)
        if self.freeze_vpt_encoder and \
            self.vpt_encoder is not None:
            self.vpt_encoder.requires_grad_(False)
        if self.freeze_ref_adapter and \
            self.ref_adapter is not None:
            self.ref_adapter.requires_grad_(False)
        if self.freeze_visual_decoder and \
            self.visual_decoder is not None:
            self.visual_decoder.requires_grad_(False)

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
            if self.vpt_encoder is not None:
                self.vpt_encoder.enable_input_require_grads()
            if self.ref_adapter is not None:
                self.ref_adapter.enable_input_require_grads()
            
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
        self.projector.gradient_checkpointing_enable()
        if self.vpt_encoder is not None:
            self.vpt_encoder.gradient_checkpointing_enable()
        if self.ref_adapter is not None:
            self.ref_adapter.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()
        if self.vpt_encoder is not None:
            self.vpt_encoder.gradient_checkpointing_disable()
        if self.ref_adapter is not None:
            self.ref_adapter.gradient_checkpointing_disable()

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
        # Step 5. REFAdapter
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'ref_adapter.' in k})
        # Step 6. Visual Decoders
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'visual_decoder.' in k})
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

    def prepare_visual_feats(self, visual_hidden_states, vpt_regions, mode='loss'):
        # get vpt feats
        if vpt_regions is not None:
            visual_prompts = self.vpt_encoder(
                visual_hidden_states,
                regions = vpt_regions, 
                return_dict = True
            )
        elif mode == 'loss':
            # fake regions for contain compute graph
            bs = visual_hidden_states.shape[0]
            w = h = int(math.sqrt(visual_hidden_states.shape[1]))
            fake_region = np.zeros((h, w))
            regions = [None] * bs
            regions[0] = [fake_region]
            vpt_count = [0] * bs
            visual_prompts = self.vpt_encoder(
                visual_hidden_states,
                regions = regions, 
                return_dict = True
            )
            visual_prompts['vpt_count'] = vpt_count
        else:
            visual_prompts = None

        # vpt feats with visual feats
        if self.vpt_encoder.config.use_projector or \
            visual_prompts is None:
            visual_feats = self.projector(visual_hidden_states)
        else:
            # concat and reuse projector
            vpt_feats = visual_prompts['vpt_feats']
            b, q, n, c = vpt_feats.shape
            _, l, _ = visual_hidden_states.shape
            vpt_feats = vpt_feats.view(b, -1, c)
            concat_feats = torch.cat([visual_hidden_states, vpt_feats], dim=1)
            concat_feats = self.projector(concat_feats)

            visual_feats = concat_feats[:, :l, :]
            vpt_feats = concat_feats[:, l:, :]
            vpt_feats = vpt_feats.view(b, q, n, vpt_feats.shape[-1])
            visual_prompts['vpt_feats'] = vpt_feats
        
        return visual_feats, visual_prompts
    
    def prepare_decode_feats(self, hidden_states, metas, shift=False, mode='loss'):
        '''
        return: 
            all_phrase_feats: List[List[Tensor]], len(phrase_feats) = batch_size, len(phrase_feats[0]) = 'num of groups in batch'
            all_unit_feats: List[List[Tensor]]
            all_ref_feats: List[List[Tensor]]
        '''

        def _match_token_pairs(start_indices, end_indices):
            # get start and end placeholder pairs
            pairs = []
            stack = []
            combined = [(index, 'start') for index in start_indices] + \
                [(index, 'end') for index in end_indices]
            combined.sort()

            for index, type_ in combined:
                if type_ == 'start':
                    stack.append(index)
                elif type_ == 'end':
                    if stack:
                        st_index = stack.pop()
                        pairs.append((st_index, index))
            return pairs

        token_masks = metas.get('token_masks', None)
        all_phrase_feats = []
        all_unit_feats = []
        all_ref_feats = []
        all_decode_groups = []
        for batch_idx, feats in enumerate(hidden_states):
            token_mask = token_masks[batch_idx]
            # ref token hidden states
            ref_indices = torch.where(token_mask == TOKEN_MASK_IDS['ref_masks'])[0].tolist()
            if len(ref_indices) == 0:
                empty_feats = feats[0:0, :]
                all_phrase_feats.append([empty_feats])
                all_unit_feats.append([empty_feats])
                all_ref_feats.append([empty_feats])
                continue

            # phrase token hidden states
            bop_indices = torch.where(token_mask == TOKEN_MASK_IDS['bop_masks'])[0].tolist()
            eop_indices = torch.where(token_mask == TOKEN_MASK_IDS['eop_masks'])[0].tolist()
            phrase_pairs = _match_token_pairs(bop_indices, eop_indices)

            # unit token hidden states
            bou_indices = torch.where(token_mask == TOKEN_MASK_IDS['bou_masks'])[0].tolist()
            eou_indices = torch.where(token_mask == TOKEN_MASK_IDS['eou_masks'])[0].tolist()
            unit_pairs = _match_token_pairs(bou_indices, eou_indices)

            if mode == 'loss':
                # not equal and not cutoff situation
                if (len(bop_indices) != len(eop_indices)) and (len(bop_indices) - 1 != len(eop_indices)):
                    print_log(f"Warning: Error Phrase indices num in batch_idx = {batch_idx}, "
                              f"BOP num = {len(bop_indices)}, EOP num = {len(eop_indices)}")
                if (len(bou_indices) != len(eou_indices)) and (len(bou_indices) - 1 != len(eou_indices)):
                    print_log(f"Warning: Error Unit indices num in batch_idx = {batch_idx}, "
                              f"BOU num = {len(bop_indices)}, EOU num = {len(eop_indices)}")

            # match 'phrase-unit-ref(s)' pairs
            # first, sort all indices
            combined_indices = [(pair[1], pair[0], 'phrase') for pair in phrase_pairs] + \
                [(pair[1], pair[0], 'unit') for pair in unit_pairs] + \
                [(idx, -1, 'ref') for idx in ref_indices]
            combined_indices.sort(reverse=True)
            # second, init 3 pointers
            p_pointers = []
            u_pointers = []
            r_pointers = []
            for pointer, (_, _, type_) in enumerate(combined_indices):
                if type_ == 'phrase':
                    p_pointers.append(pointer)
                elif type_ == 'unit':
                    u_pointers.append(pointer)
                elif type_ == 'ref':
                    r_pointers.append(pointer)
            # third, 
            decode_groups_reversed = []
            cur_group = None
            def _update_cur_group(ref_index, phrase_pair=None, unit_pair=None):
                '''
                Note: All inputs are reversed
                '''
                nonlocal cur_group
                if phrase_pair is None:
                    phrase_pair = [0, 0]
                if unit_pair is None:
                    unit_pair = [0, 0]
                if cur_group is None:
                    cur_group = dict(
                    phrase_pair = phrase_pair,
                    unit_pair = unit_pair,
                    ref_index = [ref_index]
                    )
                else:
                    cur_group['phrase_pair'] = phrase_pair  
                    cur_group['unit_pair'] = unit_pair
                    cur_group['ref_index'].append(ref_index)
                return cur_group
            
            for pointer in r_pointers:
                while len(p_pointers) > 0 and \
                    (combined_indices[p_pointers[0]][0] > combined_indices[pointer][0]):
                    p_pointers.pop(0)
                while len(u_pointers) > 0 and \
                    (combined_indices[u_pointers[0]][0] > combined_indices[pointer][0]):
                    u_pointers.pop(0)

                ref_index = combined_indices[pointer][0]
                # Notice that the pointers are reversed
                # case1: no any p&u tokens after ref tokens, refs will be appended at the end.
                if len(p_pointers) == 0 and len(u_pointers) == 0:
                    cur_group = _update_cur_group(ref_index=ref_index)
                    continue
                # case2: no any p tokens after ref tokens, refs will be appended when next pointer is u_pointer
                elif len(p_pointers) == 0 and len(u_pointers) != 0:
                    unit_pair = combined_indices[u_pointers[0]][:-1]
                    cur_group = _update_cur_group(
                        unit_pair=unit_pair,
                        ref_index=ref_index
                    )

                    if (pointer + 1) == u_pointers[0]:
                        decode_groups_reversed.append(cur_group)
                        cur_group = None
                # case3: no any u tokens after ref tokens, refs will be appended when next pointer is p_pointer
                elif len(u_pointers) == 0 and len(p_pointers) != 0:
                    phrase_pair = combined_indices[p_pointers[0]][:-1]
                    cur_group = _update_cur_group(
                        phrase_pair=phrase_pair,
                        ref_index=ref_index
                    )

                    if (pointer + 1) == p_pointers[0]:
                        decode_groups_reversed.append(cur_group)
                        cur_group = None
                # case4: there are p&u tokens after ref tokens
                else:
                    if (pointer + 1) == u_pointers[0]:
                        unit_pair = combined_indices[u_pointers[0]][:-1]
                        # case4.1: standard form, phrase-unit-refs
                        if (pointer + 2) == p_pointers[0]:
                            phrase_pair = combined_indices[p_pointers[0]][:-1]
                            cur_group = _update_cur_group(
                                phrase_pair = phrase_pair,
                                unit_pair = unit_pair,
                                ref_index = ref_index
                            )
                        # case4.2: unit-refs
                        else:
                            cur_group = _update_cur_group(
                                unit_pair=unit_pair,
                                ref_index=ref_index
                            )
                        decode_groups_reversed.append(cur_group)
                        cur_group = None
                    # case4.3: next pointer is p_pointer, phrase-refs
                    elif (pointer + 1) == p_pointers[0]:
                        phrase_pair = combined_indices[p_pointers[0]][:-1]
                        cur_group = _update_cur_group(
                            phrase_pair=phrase_pair,
                            ref_index=ref_index
                        )

                        decode_groups_reversed.append(cur_group)
                        cur_group = None
                    # case4.4: next token is ref
                    else:
                        cur_group = _update_cur_group(ref_index=ref_index)

            # last pointer
            if cur_group is not None:
                decode_groups_reversed.append(cur_group)

            # finally, reverse back the goup
            for group in decode_groups_reversed:
                for key, value in group.items():
                    group[key] = value[::-1]
            decode_groups = decode_groups_reversed[::-1]
            
            phrase_feats = []
            unit_feats = []
            ref_feats = []
            for group in decode_groups:
                p_start, p_end = group['phrase_pair']
                u_start, u_end = group['unit_pair']
                ref_index = group['ref_index']
                
                if shift:
                    p_start -= 1
                    p_end -= 1
                    u_start -= 1
                    u_end -= 1
                    ref_index = [idx - 1 for idx in ref_index]

                # non-phrase pairs, init as [0, 0]
                if p_start == p_end:
                    p_start = 0
                    p_end = p_start - 1
                # non-unit pairs, init as [0, 0]
                if u_start == u_end: 
                    u_start = 0
                    u_end = u_start - 1
                
                phrase_feats.append(feats[p_start:p_end+1, :])
                unit_feats.append(feats[u_start:u_end+1, :])
                ref_feats.append(feats[ref_index, :])
        
            all_phrase_feats.append(phrase_feats)
            all_unit_feats.append(unit_feats)
            all_ref_feats.append(ref_feats)
            all_decode_groups.append(decode_groups)

        return dict(
            phrase_feats = all_phrase_feats,
            unit_feats = all_unit_feats,
            ref_feats = all_ref_feats,
            decode_groups = all_decode_groups
        )
    
    def prepare_ref_feats(self, hidden_states, metas, mode='loss'):
        token_masks = metas.get('token_masks', None)
        ref_masks = token_masks == TOKEN_MASK_IDS['ref_masks']
        if ref_masks.sum() == 0 and mode != 'loss':
            return None, None

        max_num_queries = 0
        if self.visual_decoder is not None:
            decoder_num_queries = []
            for decoder in self.visual_decoder.values():
                decoder_num_queries.append(decoder.config.num_queries)
            max_num_queries = max(decoder_num_queries)

        if max_num_queries == 0:
            return None, None

        batch_size, _, dim_feats = hidden_states.shape
        ref_hidden_states = torch.zeros((batch_size, max_num_queries, dim_feats),
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
        ref_attention_masks = torch.zeros((batch_size, max_num_queries),
                                    dtype=torch.bool,
                                    device=hidden_states.device)

        for batch_idx, (feats, mask) in enumerate(zip(hidden_states, ref_masks)):
            ref_feats = feats[mask, :]
            ref_feats_len = ref_feats.shape[0]
            valid_len = min(max_num_queries, ref_feats_len)
            ref_hidden_states[batch_idx, :valid_len, :] = ref_feats[:valid_len,:]
            ref_attention_masks[batch_idx, :valid_len] = True
        return ref_hidden_states, ref_attention_masks

    def prepare_token_masks(self, ids):
        mask_tokens = dict(
            ref_masks = VISUAL_REFERENCE_TOKEN,
            bou_masks = BOU_TOKEN,
            eou_masks = EOU_TOKEN,
            bop_masks = PHRASE_ST_PLACEHOLDER_STAGE2,
            eop_masks = PHRASE_ED_PLACEHOLDER_STAGE2
        )
        token_masks = torch.zeros_like(ids, dtype=torch.uint8)
        for batch_idx, ids in enumerate(ids):
            for type, token in mask_tokens.items():
                mask = ids == self.token_ids[token]
                token_masks[batch_idx, mask] = TOKEN_MASK_IDS[type]
        return token_masks

    def forward(self, data, data_samples=None, mode='loss'):
        metas = dict()
        meta_keys = [
            # 'pixel_values', 'image_path',
            # 'ori_height', 'ori_width',
            'decode_labels', 'decode_units',
            'decode_seqs', 'conversations',
            'pixel_masks'
        ]
        assert 'pixel_values' in data, "pixel_values must in data dict."
        for key in meta_keys:
            metas[key] = data.get(key, None)

        visual_outputs = self.visual_encoder(
            data['pixel_values'].to(self.visual_encoder.dtype),
            output_hidden_states=True)
        selected_feats = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]

        if self.visual_tower is None:
            metas['visual_hidden_states'] = [feats[:, 1:] for feats in visual_outputs.hidden_states]
        else:
            visual_tower_outputs = self.visual_tower(data['pixel_values_tower'].to(self.visual_tower.dtype))
            metas['visual_hidden_states'] = visual_tower_outputs['hidden_states']
            metas['visual_hidden_states'].append(selected_feats)

        if self.vpt_encoder is None:
            pixel_values = self.projector(selected_feats)
            data['pixel_values'] = pixel_values
        else:
            vpt_regions = data.get('visual_prompts', None)
            visual_feats, visual_prompts = self.prepare_visual_feats(
                selected_feats, 
                vpt_regions, 
                mode
            )
            if visual_prompts is not None:
                data.update(visual_prompts) 
            data['pixel_values'] = visual_feats

        # prepare data for train/predict
        try:
            if mode == 'loss':
                data['token_masks'] = self.prepare_token_masks(data['input_ids'])
            
            if mode == 'predict':
                labels_mask = (data['labels'].detach().cpu().numpy()[0] == IGNORE_INDEX).tolist()
                trim = ['input_ids']
                remove = ['labels', 'attention_mask', 'position_ids']
                for key in trim:
                    value = data.get(key, None)
                    if value is None: continue
                    data[key] = data[key][0][:labels_mask.index(False)].unsqueeze(0)
                for key in remove:
                    data[key] = None

            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
            if self.cutoff_len is not None:
                for key, value in data.items():
                    if value is None: continue
                    if value.shape[1] > self.cutoff_len:
                        data[key] = value[:, :self.cutoff_len]
            
            if mode == 'loss':
                metas['token_masks'] = data.pop('token_masks')
        except Exception as e:
            print(e)
            file_prefix = f"wrong_prepare_data"
            save_wrong_data(file_prefix, data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples, metas)
        elif mode == 'predict':
            return self.predict(data, data_samples, metas)
        elif mode == 'tensor':
            return self._forward(data, data_samples, metas)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None, metas=None):

        outputs = self.llm(**data)

        return outputs
    
    def modules_forward_pipeline(self, hidden_states, metas, mode):
        results = dict()
        if self.visual_decoder is None:
            return results

        decode_feats = self.prepare_decode_feats(
            hidden_states,
            metas=metas,
            shift=True,
            mode=mode
        )
        results['decode_groups'] = decode_feats['decode_groups']
        if self.ref_adapter is not None:

            # all empty features
            if mode == 'predict' and all([len(feats) == 1 and feats[0].shape[0] == 0 \
                                          for feats in decode_feats['ref_feats']]):
                return results
            
            ref_outputs = self.ref_adapter(
                decode_feats['phrase_feats'],
                decode_feats['unit_feats'],
                decode_feats['ref_feats'],
                metas=metas,
                mode=mode
            )
            ref_hidden_states = ref_outputs['ref_hidden_states']
            ref_mask = ref_outputs['ref_mask']
        else:
            ref_hidden_states, ref_mask = self.prepare_ref_feats(
                hidden_states,
                metas=metas,
                mode=mode
            )
            if mode == 'predict' and ref_hidden_states is None:
                return results
        
        # decoders
        if self.visual_decoder is not None:
            visual_decoder_outputs = dict()
            visual_hidden_states = metas['visual_hidden_states']

            for type, decoder in self.visual_decoder.items():
                decode_outputs = decoder(
                    visual_hidden_states,
                    ref_hidden_states,
                    visual_mask=None,
                    ref_mask=ref_mask,
                    metas=metas,
                    mode=mode
                )
                visual_decoder_outputs[type] = decode_outputs
            results['visual_decoder'] = visual_decoder_outputs
        
        return results

    @torch.no_grad()
    def predict(self, data, data_samples=None, metas=None):
        def _pad_token_masks(token_masks, target_size, pad_side):
            if token_masks is not None:
                dtype = token_masks.dtype
                device = token_masks.device
                padded_token_masks = torch.zeros(
                    target_size).to(dtype).to(device)
                cur_length = token_masks.shape[1]
                if pad_side == 'left':
                    padded_token_masks[:, -cur_length:] = token_masks
                elif pad_side == 'right':
                    padded_token_masks[:, :cur_length] = token_masks
                else:
                    raise ValueError('only support left or right side')
            else:
                padded_token_masks = None
            return padded_token_masks

        for key in data.keys():
            if data[key] is not None:
                data[key] = data[key].to(self.llm.dtype)
        
        llm_outputs = self.llm.generate(
                **data,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria)
        
        # prepare hidden_states for modules
        hidden_states = []
        for time_step_feats in llm_outputs.hidden_states:
            last_layer_feats = time_step_feats[-1]
            hidden_states.append(last_layer_feats)
        hidden_states = torch.cat(hidden_states, dim=1)

        # prepare special token masks for modules
        # remove last generated token (the last one usually is </s>)
        # in all situations, the last token will not used for inference (</s> or cutoff as max_length)
        # pad left because no decode token in prompts
        token_masks = self.prepare_token_masks(llm_outputs.sequences[:, :-1])
        metas['token_masks'] = _pad_token_masks(
            token_masks=token_masks,
            target_size=hidden_states.shape[:-1],
            pad_side='left'
        )

        # get pipeline outputs
        pipeline_outputs = self.modules_forward_pipeline(
            hidden_states=hidden_states, 
            metas=metas, 
            mode='predict'
        )
        decode_groups = pipeline_outputs.get('decode_groups', None)
        decoder_outputs = pipeline_outputs.get('visual_decoder', None)

        results = []
        for batch_idx, generate_id in enumerate(llm_outputs.sequences):
            # remove padding length in decode_groups
            prompt_length = llm_outputs.hidden_states[0][-1].shape[1]
            decode_group = []
            if (decode_groups is not None) and (len(decode_groups) != 0):
                for group in decode_groups[batch_idx]:
                    for key, value in group.items():
                        group[key] = [idx - prompt_length for idx in value]
                    decode_group.append(group)
                decode_groups = decode_groups if len(decode_groups) > 0 else None
            
            # decoder outputs
            decoder_output = dict()
            if decoder_outputs is not None:
                for type, output in decoder_outputs.items():
                    decoder_output[type] = output['preds'][batch_idx] \
                        if output is not None else None
        
            decoder_output = decoder_output if len(decoder_output) > 0 else None

            results.append(
                {
                    'generate_ids': generate_id,
                    'decode_groups': decode_group,
                    'decoder_outputs': decoder_output
                }
            )

        return results

    def compute_loss_llm(self, logits, labels):
        cot_weight = 1
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        weight = torch.ones_like(shift_labels).to(shift_logits.dtype)
        idx = torch.zeros_like(shift_labels)
        idx[shift_labels!=IGNORE_INDEX] = 2
        bs = shift_labels.shape[0]
        for batch_idx in range(bs):
            # CoT chunks
            bot_id = self.token_ids[BOT_TOKEN]
            eot_id = self.token_ids[EOT_TOKEN]
            cot_start = torch.where(shift_labels[batch_idx] == bot_id)[0].tolist()
            cot_end = torch.where(shift_labels[batch_idx] == eot_id)[0].tolist()                
            if len(cot_start) == len(cot_end):
                for st, ed in zip(cot_start, cot_end):
                    weight[batch_idx, st:ed+1] = cot_weight
                    idx[batch_idx, st:ed+1] = 1
            elif len(cot_start) == len(cot_end) + 1:
                last_st = cot_start[-1]
                for st, ed in zip(cot_start, cot_end):
                    weight[batch_idx, st:ed+1] = cot_weight
                    idx[batch_idx, st:ed+1] = 1
                weight[batch_idx, last_st:] = cot_weight
                idx[batch_idx, last_st:] = 1
            else:
                print("<Task> and </Task> not match!")
                file_prefix = f"wrong_cot"
                save_wrong_data(file_prefix,shift_labels.clone().detach().cpu().numpy())
            
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        weight = weight.view(-1)

        idx = idx.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        weight = weight.to(shift_logits.dtype).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        weighted_loss = weight[shift_labels!=IGNORE_INDEX] * loss[shift_labels!=IGNORE_INDEX]
        cot_loss = weight[idx==1] * loss[idx==1]
        answer_loss = weight[idx==2] * loss[idx==2]
        return weighted_loss.mean(), cot_loss.mean(), answer_loss.mean()

    def compute_loss(self, data, data_samples=None, metas=None):
        labels = data.pop("labels") # to avoid computing loss in llm_base_class.forward()
        if 'output_hidden_states' not in data.keys():
            data['output_hidden_states'] = True
        
        loss_dict = dict() # for loss backward propagation
        cost_dict = dict() # for record only
        # llm loss
        outputs = self.llm(**data)
        logits = outputs.logits
        loss_llm, loss_cot, loss_answer = self.compute_loss_llm(logits, labels)
        loss_dict['llm'] = loss_llm
        cost_dict['llm_cost'] = loss_llm
        cost_dict['cot_cost'] = loss_cot
        cost_dict['answer_cost'] = loss_answer

        # all modules forward
        modules_outputs = self.modules_forward_pipeline(
            hidden_states=outputs.hidden_states[-1], 
            metas=metas, 
            mode='loss'
        )
        decoder_outputs = modules_outputs.get('visual_decoder', None)
        if decoder_outputs is not None:
            for type, outputs in decoder_outputs.items():
                loss_dict[type] = outputs['loss']
                cost_dict[f'{type}_cost'] = outputs['loss']

        # loss
        loss = 0
        for key, value in loss_dict.items():
            coefficient = self.loss_coefficient[key]
            loss += coefficient * value
        
        loss_result = dict()
        loss_result['loss'] = loss
        loss_result.update(cost_dict)
        return loss_result

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

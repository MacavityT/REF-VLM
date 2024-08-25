# Copyright (c) OpenMMLab. All rights reserved.
import math
import redis
import random
from collections import OrderedDict
import json
import jsonlines
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.logging import print_log
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig,GenerationConfig, StoppingCriteriaList

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from .modules import *
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad, traverse_dict,
                    prepare_inputs_labels_for_multimodal,
                    save_wrong_data)
from xtuner.utils.constants import (
    SPECIAL_TOKENS,
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN,
    IMAGE_TOKEN_INDEX,
    VISUAL_REFERENCE_TOKEN,
    VISUAL_REPRESENTATION_TOKEN,
)
from xtuner.tools.utils import get_random_available_port
from torch.nn import CrossEntropyLoss, MSELoss

DECODER_CONFIG_CLASS = {
    'box': BoxDecoderConfig,
    'mask': MaskDecoderConfig
}

DECODER_MODEL_CLASS = {
    'box': BoxDecoderModel,
    'mask': MaskDecoderModel
}

class OkapiModel(BaseModel):

    def __init__(self,
                 llm,
                 tokenizer=None,
                 visual_encoder=None,
                 vpt_encoder=None,
                 projector=None,
                 visual_sync_tuner=None,
                 moe_adapter=None,
                 visual_decoder=None,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_projector=False,
                 freeze_vpt_encoder=False,
                 freeze_visual_sync_tuner=False,
                 freeze_moe_adapter=False,
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
        self.freeze_visual_sync_tuner = freeze_visual_sync_tuner
        self.freeze_moe_adapter = freeze_moe_adapter
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
                VISUAL_REPRESENTATION_TOKEN,
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
            if visual_sync_tuner is not None and 'type' in visual_sync_tuner:
                self.visual_sync_tuner = self._build_from_cfg_or_module(
                    visual_sync_tuner).to(self.llm.dtype)
            else:
                self.visual_sync_tuner = None
            if moe_adapter is not None and 'type' in moe_adapter:
                self.moe_adapter = self._build_from_cfg_or_module(
                    moe_adapter).to(self.llm.dtype)
            else:
                self.moe_adapter = None

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
        if self.visual_sync_tuner is None and visual_sync_tuner is not None:
            sync_tuner_config = SyncTunerConfig(**visual_sync_tuner)
            assert sync_tuner_config.num_queries > 0, 'vrt length error!'
            self.visual_sync_tuner = SyncTunerModel(sync_tuner_config).to(
                self.llm.dtype)
        if moe_adapter is not None and \
            'type' not in moe_adapter:
            moe_adapter_config = MoEAdapterConfig(**moe_adapter)
            self.moe_adapter = MoEAdapterModel(moe_adapter_config).to(
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
        if self.freeze_visual_sync_tuner and \
            self.visual_sync_tuner is not None:
            self.visual_sync_tuner.requires_grad_(False)
        if self.freeze_moe_adapter and \
            self.moe_adapter is not None:
            self.moe_adapter.requires_grad_(False)
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
            if self.visual_sync_tuner is not None:
                self.visual_sync_tuner.enable_input_require_grads()
            if self.moe_adapter is not None:
                self.moe_adapter.enable_input_require_grads()
            
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
        if self.visual_sync_tuner is not None:
            self.visual_sync_tuner.gradient_checkpointing_enable()
        if self.moe_adapter is not None:
            self.moe_adapter.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()
        if self.vpt_encoder is not None:
            self.vpt_encoder.gradient_checkpointing_disable()
        if self.visual_sync_tuner is not None:
            self.visual_sync_tuner.gradient_checkpointing_disable()
        if self.moe_adapter is not None:
            self.moe_adapter.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    #taiyan TODO:  check correctness
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
        # Step 5. Visual SyncTuner
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'visual_sync_tuner.' in k})
        # Step 6. MoEAdapter
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'moe_adapter.' in k})
        # Step 7. Visual Decoders
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
    
    def prepare_vrt_feats(self, hidden_states, metas, mode='loss'):
        vrt_masks = metas.get('vrt_masks', None)
        if vrt_masks.sum() == 0 and mode != 'loss':
            return None
        
        vrt_hidden_states = []
        num_queries = self.visual_sync_tuner.config.num_queries
        dim_feats = hidden_states.shape[-1]
        for feats, mask in zip(hidden_states, vrt_masks):
            vrt_feats = feats[mask, :]
            if vrt_feats.shape[0] > 0:
                assert vrt_feats.shape[0] == num_queries
                vrt_hidden_states.append(vrt_feats)
            else:
                # prepare fake features for all mode (for locating batch features in prediction mode)
                fake_feats = torch.zeros(
                    num_queries, 
                    dim_feats
                    ).to(hidden_states.device).to(hidden_states.dtype)
                vrt_hidden_states.append(fake_feats)
        vrt_hidden_states = torch.stack(vrt_hidden_states)
        return vrt_hidden_states

    def prepare_ref_feats(self, hidden_states, metas, mode='loss'):
        ref_masks = metas.get('ref_masks', None)
        if ref_masks.sum() == 0 and mode != 'loss':
            return None, None

        moe_num_queries = 0
        if self.moe_adapter is not None:
            moe_num_queries = self.moe_adapter.config.num_queries

        max_decoder_num_queries = 0
        if self.visual_decoder is not None:
            decoder_num_queries = []
            for decoder in self.visual_decoder.values():
                decoder_num_queries.append(decoder.config.num_queries)
            max_decoder_num_queries = max(decoder_num_queries)
            if moe_num_queries != 0:
                assert max_decoder_num_queries <= moe_num_queries

        max_num_queries = max(moe_num_queries, max_decoder_num_queries)
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
            ref_hidden_states[batch_idx, :ref_feats_len, :] = ref_feats
            ref_attention_masks[batch_idx, :ref_feats_len] = True
        return ref_hidden_states, ref_attention_masks

    def forward(self, data, data_samples=None, mode='loss'):
        metas = dict()
        meta_keys = [
            'ori_image', 'image_path',
            'ori_height', 'ori_width',
            'decode_labels', 'decode_units',
            'decode_seqs', 'conversations',
            'pixel_masks'
        ]
        assert 'pixel_values' in data, "pixel_values must in data dict."
        for key in meta_keys:
            if key == 'ori_image':
                metas[key] = data.get('pixel_values', None)
            else:
                metas[key] = data.get(key, None)

        visual_outputs = self.visual_encoder(
            data['pixel_values'].to(self.visual_encoder.dtype),
            output_hidden_states=True)
        selected_feats = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        metas['visual_hidden_states'] = [feats[:, 1:] for feats in visual_outputs.hidden_states]

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
            # vrt and ref mask
            vrt_masks = []
            ref_masks = []
            input_ids = data['input_ids']
            for ids in input_ids:
                vrt_mask = ids == self.token_ids[VISUAL_REPRESENTATION_TOKEN]
                ref_mask = ids == self.token_ids[VISUAL_REFERENCE_TOKEN]
                vrt_masks.append(vrt_mask.bool())
                ref_masks.append(ref_mask.bool())
            data['vrt_masks'] = vrt_masks
            data['ref_masks'] = ref_masks

            if mode == 'predict':
                labels_mask = (data['labels'].detach().cpu().numpy()[0] == IGNORE_INDEX).tolist()
                trim = ['input_ids', 'vrt_masks', 'ref_masks']
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
            
            metas['vrt_masks'] = data.pop('vrt_masks')
            metas['ref_masks'] = data.pop('ref_masks')
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

        ref_hidden_states, ref_attention_masks = None, None
        ref_used_module = [self.moe_adapter, self.visual_decoder]
        if any(module is not None for module in ref_used_module):
            ref_hidden_states, ref_attention_masks = self.prepare_ref_feats(
                hidden_states,
                metas=metas,
                mode=mode
            )
        if ref_hidden_states is None:
            # only in predict process
            return results

        # visual sync tuner
        rec_outputs = None
        if self.visual_sync_tuner is not None:
            vrt_hidden_states = self.prepare_vrt_feats(
                hidden_states=hidden_states,
                metas=metas,
                mode=mode
            )
            rec_outputs = self.visual_sync_tuner(
                vrt_hidden_states,
                metas=metas,
                mode=mode
            )
            results['visual_sync_tuner'] = rec_outputs

        # moe adapter
        moe_outputs = None
        if self.moe_adapter is not None:
            moe_outputs = self.moe_adapter(
                ref_hidden_states,
                # ref_attention_masks,
                mode=mode
            )
            results['moe_adapter'] = moe_outputs
        
        # decoders
        if self.visual_decoder is not None:
            visual_decoder_outputs = dict()
            if rec_outputs is None:
                visual_hidden_states = metas['visual_hidden_states']
            else:
                # pyramid outputs
                visual_hidden_states = rec_outputs['hidden_states']

            if moe_outputs is None:
                decode_hidden_states = ref_hidden_states
            else:
                # last layer output
                decode_hidden_states = moe_outputs['hidden_states'][-1]
            
            for type, decoder in self.visual_decoder.items():
                decode_outputs = decoder(
                    visual_hidden_states,
                    decode_hidden_states,
                    visual_mask=None,
                    ref_mask=ref_attention_masks,
                    metas=metas,
                    mode=mode
                )
                visual_decoder_outputs[type] = decode_outputs
            results['visual_decoder'] = visual_decoder_outputs
        
        return results

    @torch.no_grad()
    def predict(self, data, data_samples=None, metas=None):
        def _pad_special_token_masks(special_token_masks, target_size, pad_side):
            if special_token_masks is not None:
                dtype = special_token_masks.dtype
                device = special_token_masks.device
                padded_special_token_masks = torch.zeros(
                    target_size).to(dtype).to(device)
                cur_length = special_token_masks.shape[1]
                if pad_side == 'left':
                    padded_special_token_masks[:, :cur_length] = special_token_masks
                elif pad_side == 'right':
                    padded_special_token_masks[:, -cur_length:] = special_token_masks
                else:
                    raise ValueError('only support left or right side')
            else:
                padded_special_token_masks = None
            return padded_special_token_masks

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
        vrt_masks = metas.get('vrt_masks', None)
        metas['vrt_masks'] = _pad_special_token_masks(
            special_token_masks=vrt_masks,
            target_size=hidden_states.shape[:-1],
            pad_side='left'
        )

        # remove last generated token (the last one usually is </s>)
        # in all situations, the last token will not used for inference (</s> or cutoff as max_length)
        ref_masks = llm_outputs.sequences[:, :-1] == \
            self.token_ids[VISUAL_REFERENCE_TOKEN]
        metas['ref_masks'] = _pad_special_token_masks(
            special_token_masks=ref_masks,
            target_size=hidden_states.shape[:-1],
            pad_side='right'
        )

        # get pipeline outputs
        pipeline_outputs = self.modules_forward_pipeline(
            hidden_states=hidden_states, 
            metas=metas, 
            mode='predict'
        )
        sync_tuner_outputs = pipeline_outputs.get('visual_sync_tuner', None)
        decoder_outputs = pipeline_outputs.get('visual_decoder', None)

        results = dict()
        results['generate_ids'] = llm_outputs.sequences  # [batch,token_id_length]
        # {'preds': Tensor}
        results['sync_tuner_outputs'] = sync_tuner_outputs
        # {'box':{'loss':, 'preds':(List[Tensor])[batch]}, 'mask':(List[Tensor])}
        results['decoder_outputs'] = decoder_outputs 
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
        rec_outputs = modules_outputs.get('visual_sync_tuner', None)
        if rec_outputs is not None:
            loss_dict['rec'] = rec_outputs['loss']
            cost_dict['rec_cost'] = rec_outputs['loss']
        
        moe_outputs = modules_outputs.get('moe_adapter', None)
        if moe_outputs is not None:
            loss_dict['moe'] = moe_outputs['loss']
            cost_dict['moe_cost'] = moe_outputs['loss']
        
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

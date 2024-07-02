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
from .modules import (
    ProjectorConfig, ProjectorModel,
    VPTEncoderConfig, VPTEncoderModel,
    SyncTunerConfig, SyncTunerModel,
    dispatch_modules
    )
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad, traverse_dict,
                    prepare_inputs_labels_for_multimodal,
                    save_wrong_data)
from xtuner.utils.constants import (
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN,
    IMAGE_TOKEN_INDEX,
    SPECIAL_TOKENS
)
from xtuner.tools.utils import get_random_available_port
from torch.nn import CrossEntropyLoss, MSELoss

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
                 freeze_visual_sync_tuner=False,
                 freeze_moe_adapter=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 cutoff_len=None,
                 max_position_embeddings=None,
                 loss_weight=None,
                 cot_weight=None,
                 vrt_weight=None,
                 image_pool=None):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_projector = freeze_projector
        self.freeze_visual_sync_tuner = freeze_visual_sync_tuner
        self.freeze_moe_adapter = freeze_moe_adapter
        self.cutoff_len = cutoff_len
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
                BOV_TOKEN, EOV_TOKEN
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
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.eos_token_id)
            self.max_new_tokens = 512
            self.gen_config = GenerationConfig(**default_generation_kwargs)
            self.stop_criteria = StoppingCriteriaList()

            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
            if vpt_encoder is not None:
                self.vpt_encoder = self._build_from_cfg_or_module(
                    vpt_encoder)
            if projector is not None:
                self.projector = self._build_from_cfg_or_module(
                    projector)
            if visual_sync_tuner is not None and 'type' in visual_sync_tuner:
                self.visual_sync_tuner = self._build_from_cfg_or_module(
                    visual_sync_tuner)
            else:
                self.visual_sync_tuner = None
            if moe_adapter is not None and 'type' in moe_adapter:
                self.moe_adapter = self._build_from_cfg_or_module(
                    moe_adapter)
            else:
                self.moe_adapter = None
            # visual_decoder
            
        # # self.image_pool = redis.StrictRedis(host='localhost', port=6379, db=0)
        # if image_pool is not None:
        #     with jsonlines.open(image_pool, 'r') as f:
        #         self.image_pool_idx2path = [data for data in f]
        #         self.image_pool_path2idx = {path: index for index, path in \
        #             enumerate(self.image_pool_idx2path)}
        # else:
        #     self.image_pool_idx2path = None
        #     self.image_pool_path2idx = None
        
        # self.register_buffer(
        #     "image_pool_used_idx", 
        #     torch.zeros(len(self.image_pool_idx2path)).bool(), 
        #     persistent=False
        # )

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
                strategy='pooling',
                vis_feats_len=vis_feats_len,
                mask_patch_len=vis_feats_len // num_patches,
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                # llm_hidden_size=self.llm.config.hidden_size,
                # depth=projector_depth
            )
            self.vpt_encoder = VPTEncoderModel(vpt_encoder_config).to(
                self.visual_encoder.dtype)
        if visual_sync_tuner is not None:
            sync_tuner_config = SyncTunerConfig(**visual_sync_tuner)
            self.visual_sync_tuner = SyncTunerModel(sync_tuner_config).to(
                self.visual_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)
        if self.freeze_projector:
            self.projector.requires_grad_(False)
        if self.freeze_visual_sync_tuner and \
            self.visual_sync_tuner is not None:
            self.visual_sync_tuner.requires_grad_(False)
        if self.freeze_moe_adapter and \
            self.moe_adapter is not None:
            self.moe_adapter.requires_grad_(False)

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

        self.cot_weight = cot_weight if cot_weight is not None else 1
        self.vrt_weight = vrt_weight if vrt_weight is not None else 1
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
        metas = dict()
        if 'pixel_values' in data:
            metas['ori_image'] = data['pixel_values']
            metas['image_path'] = data['image_path']
            metas['ori_height'] = data['ori_height']
            metas['ori_width'] = data['ori_width']

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
            else:
                # fake regions for contain compute graph
                bs = selected_feats.shape[0]
                w = h = int(math.sqrt(selected_feats.shape[1]))
                fake_region = np.zeros((h, w))
                regions = [None] * bs
                regions[0] = [fake_region]
                vpt_count = [0] * bs
                visual_prompts = self.vpt_encoder(
                    selected_feats,
                    regions = regions, 
                    return_dict = True
                )
                visual_prompts['vpt_count'] = vpt_count

            # pixel_values = self.projector(selected_feats)
            # data['pixel_values'] = pixel_values
            # data.update(visual_prompts)

            # concat and reuse projector
            vpt_feats = visual_prompts['vpt_feats']
            b, q, n, c = vpt_feats.shape
            _, l, _ = selected_feats.shape
            vpt_feats = vpt_feats.view(b, -1, c)
            concat_feats = torch.cat([selected_feats, vpt_feats], dim=1)
            concat_feats = self.projector(concat_feats)

            pixel_values = concat_feats[:, :l, :]
            vpt_feats = concat_feats[:, l:, :]
            vpt_feats = vpt_feats.view(b, q, n, vpt_feats.shape[-1])
            visual_prompts['vpt_feats'] = vpt_feats
            data.update(visual_prompts)
            data['pixel_values'] = pixel_values

            # get vrt mask
            bov_id = self.token_ids[BOV_TOKEN]
            eov_id = self.token_ids[EOV_TOKEN]
            input_ids = data['input_ids']
            bov_indices = []
            eov_indices = []
            for ids in input_ids:
                bov_idx = torch.where(ids == bov_id)[0].tolist()
                eov_idx = torch.where(ids == eov_id)[0].tolist()
                img_idx = torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist()
                assert len(bov_idx) == len(eov_idx)
                if len(bov_idx) == 1:
                    # TODO: double check vrt tokens position
                    assert len(img_idx) == 1 and img_idx < bov_idx
                    image_token_len = pixel_values.shape[1]
                    bov_indices.append(bov_idx[0] + image_token_len - 1)
                    eov_indices.append(eov_idx[0] + image_token_len - 1)
                elif len(bov_idx) == 0:
                    bov_indices.append(-1)
                    eov_indices.append(-1)
                else:
                    raise ValueError("vrt start num should be 0 or 1.")
            metas['bov_indices'] = bov_indices
            metas['eov_indices'] = eov_indices
            
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
    
    # TODO： Aaron add
    def predict(self, data, data_samples=None, metas=None):
        for key in data.keys():
            if data[key] is not None:
                data[key] = data[key].to(self.llm.dtype)
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

    def compute_loss_llm(self, logits, labels):
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
                    weight[batch_idx, st:ed+1] = self.cot_weight
                    idx[batch_idx, st:ed+1] = 1
            elif len(cot_start) == len(cot_end) + 1:
                last_st = cot_start[-1]
                for st, ed in zip(cot_start, cot_end):
                    weight[batch_idx, st:ed+1] = self.cot_weight
                    idx[batch_idx, st:ed+1] = 1
                weight[batch_idx, last_st:] = self.cot_weight
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

    def compute_loss_sync_tuner(self, vrt_feats, image, image_path):
        rec_flags = []
        # for path in image_path:
            # if path == '':
            #     rec_flag = False # fake image
            # # elif self.image_pool.sismember("processed_images", path):
            # elif self.image_pool_used_idx[self.image_pool_path2idx[path]]:
            #     rec_flag = False
            # else:
            #     rec_flag = np.random.uniform(0, 1) < self.visual_sync_tuner.config.ratio
            #     self.image_pool_used_idx[self.image_pool_path2idx[path]] = rec_flag
            #     # if rec_flag:
            #     #     # self.image_pool.sadd("processed_images", path)
        
        for _ in range(vrt_feats.shape[0]):
            rec_flag = np.random.uniform(0, 1) < self.visual_sync_tuner.config.ratio
            rec_flags.append(rec_flag)
        if not any(rec_flags):
            idx = random.randint(0, len(rec_flags) - 1)
            rec_flags[idx] = True

        b, c, h, w = image.shape
        mask = torch.Tensor(rec_flags).expand(c, h, w, b).permute(3, 0, 1, 2).bool() # b, c, h, w
        
        loss_fct = MSELoss(reduction='none')
        pred = self.visual_sync_tuner(vrt_feats)['last_hidden_state']

        # transform pred to target shape
        b, l, c = pred.shape
        grid_size = int(math.sqrt(l))
        if pred.shape[1] != image.shape[-1] * image.shape[-2]:
            pred = pred.reshape(b, grid_size, grid_size, c).permute(0, 3, 1, 2) # b, c, h, w
            pred = F.interpolate(
                pred,
                size=image.shape[2:],
                mode="bicubic",
                align_corners=False
            )

        # ignore loss when 'rec_flag = False'
        pred = pred[mask]
        target = image[mask].to(vrt_feats.dtype)

        # # ignore padding value
        # ignore_mask = target == 1
        # pred = pred[ignore_mask]
        # target = target[ignore_mask]

        loss_rec = loss_fct(pred, target)
        return loss_rec.mean()

    def compute_loss_decoder(self, hidden_states, targets):
        if targets is None:
            return 0
        return 0

    def compute_loss(self, data, data_samples=None, metas=None):
        labels = data.pop("labels")
        decode_labels = data.get('decode_labels', None)
        if 'output_hidden_states' not in data.keys():
            data['output_hidden_states'] = True
        
        loss_dict = dict()
        # llm 
        outputs = self.llm(**data)
        logits = outputs.logits
        loss_llm, loss_cot, loss_answer = self.compute_loss_llm(logits, labels)
        loss_dict['llm_cost'] = loss_llm
        loss_dict['cot_cost'] = loss_cot
        loss_dict['answer_cost'] = loss_answer

        # sync tuner reconstruction loss
        selected_hidden_states = outputs.hidden_states[-1]
        bov_indices = metas['bov_indices']
        eov_indices = metas['eov_indices']
        vrt_hidden_states = []
        if self.visual_sync_tuner is not None:
            for batch_idx, (bov_idx, eov_idx) in enumerate(zip(bov_indices, eov_indices)):
                if eov_idx - bov_idx - 1 == self.visual_sync_tuner.config.num_queries:
                    vrt_hidden_states.append(selected_hidden_states[batch_idx, bov_idx+1:eov_idx, :])
                elif bov_idx == -1 and eov_idx == -1:
                    # fake features
                    b, _, c = selected_hidden_states.shape
                    vrt_hidden_states.append(
                        torch.zeros(
                            b, self.visual_sync_tuner.config.num_queries, c
                            ).to(selected_hidden_states.device).to(selected_hidden_states.dtype)
                    )
                else:
                    raise ValueError(f"vrt length must equal to num_queries({self.visual_sync_tuner.config.num_queries}), but get {eov_idx - bov_idx - 1}")
            vrt_hidden_states = torch.stack(vrt_hidden_states)
            loss_rec = self.compute_loss_sync_tuner(
                vrt_feats = vrt_hidden_states,
                image = metas['ori_image'],
                image_path = metas['image_path'],
            )
            loss_dict['rec_cost'] = loss_rec
        else:
            loss_rec = 0
        
        # decode loss
        loss_decoder = self.compute_loss_decoder(
            selected_hidden_states,
            decode_labels
        )

        loss = loss_llm + loss_rec + loss_decoder
        loss_dict['loss'] = loss
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

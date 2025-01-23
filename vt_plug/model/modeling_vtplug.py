from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator
import math
import torch
import torch.nn as nn
import numpy as np
from torch import nn
from transformers import (
    GenerationMixin, 
    PreTrainedModel, 
    CLIPVisionModel, 
    CLIPModel,
    AutoModelForCausalLM,
    AutoModel,
    Qwen2ForCausalLM,
    CLIPVisionConfig,
    Qwen2Config,
    LlavaForConditionalGeneration,
    
)
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from mmengine.logging import print_log
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)


from .configuration_vtplug import VTPlugConfig
from .utils import prepare_inputs_labels_for_multimodal
from .modules import *

logger = logging.get_logger(__name__)

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


class VTPlugPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = VTPlugConfig
    base_model_prefix = "model"
    _no_split_modules = []
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True



@dataclass
# Copied from transformers.models.llava.modeling_llava.LlavaCausalLMOutputWithPast with Llava->Aria
class VTPlugCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Aria causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None




# adapted from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration
class VTPlugForConditionalGeneration(VTPlugPretrainedModel):
    """
    VT-PLUG model for conditional generation tasks.
    This model combines a vision tower, a multi-modal projector, and a language model
    to perform tasks that involve both image and text inputs.
    """

    def __init__(self, config: VTPlugConfig):
        super().__init__(config)
        # self.vision_tower = CLIPVisionModel.from_pretrained(self.config.vision_tower, trust_remote_code=True)
        # self.language_model = AutoModelForCausalLM.from_pretrained(self.config.llm_path, trust_remote_code=True)

        # if self.config.mm_projector_path is not None:
        #     self.projector = AutoModel.from_pretrained(self.config.mm_projector_path, trust_remote_code=True)
        # else:
        #     self.projector = ProjectorModel(self.config.mm_projector_config)


        # if self.config.vpt_encoder_config is not None:
        #     if self.config.vpt_encoder_path is not None:
        #         self.vpt_encoder = AutoModel.from_pretrained(self.config.vpt_encoder_path, trust_remote_code=True)
        #     else:
        #         self.vpt_encoder = VPTEncoderModel(self.config.vpt_encoder_config)
        
        # self.visual_decoder = nn.ModuleDict()
        # if self.config.decoder_config != {}:
        #     if self.config.decoder_path != {}:
        #         for key, value in self.config.decoder_path.items():
        #             self.decoder_config[key] = AutoModel.from_pretrained(value, trust_remote_code=True)
        #     else:
        #         for key, value in self.config.decoder_config.items():
        #             self.decoder_config[key] = DECODER_MODEL_CLASS[key](value)

        self.vision_tower = CLIPVisionModel(self.config.vision_config.vision_config)
        self.language_model = Qwen2ForCausalLM(self.config.text_config)
        self.projector = ProjectorModel(self.config.mm_projector_config)
        if self.config.vpt_encoder_config is not None:
            self.vpt_encoder = VPTEncoderModel(self.config.vpt_encoder_config)
        
        self.visual_decoder = nn.ModuleDict()
        if self.config.decoder_config != {}:
            for key, value in self.config.decoder_config.items():
                self.decoder_config[key] = DECODER_MODEL_CLASS[key](value)

        # generate config
        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # eos_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id
            # if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )
        self.config = config
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.stop_criteria = StoppingCriteriaList()


    def freeze_vit(self):
        """Freeze the parameters of the vision tower."""
        for param in self.vision_tower.parameters():
            param.requires_grad = False

    def freeze_projector(self):
        """Freeze the parameters of the multi-modal projector."""
        for param in self.projector.parameters():
            param.requires_grad = False

    def freeze_llm(self):
        """Freeze the parameters of the language model."""
        for param in self.language_model.parameters():
            param.requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        """Retrieve the input embeddings from the language model."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embeddings for the language model."""
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Retrieve the output embeddings from the language model."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        """Set the output embeddings for the language model."""
        self.language_model.set_output_embeddings(value)

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



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, VTPlugCausalLMOutputWithPast]:
        """
        Forward pass of the VTPLUGForConditionalGeneration model.
        This method processes both text and image inputs, merges them if necessary,
        and generates output using the language model.
        Args:
            input_ids (torch.LongTensor, optional): Input token ids.
            pixel_values (torch.FloatTensor, optional): Pixel values of the images.
            pixel_mask (torch.LongTensor, optional): Mask for the pixel values.
            attention_mask (torch.Tensor, optional): Attention mask.
            position_ids (torch.LongTensor, optional): Position ids.
            past_key_values (List[torch.FloatTensor], optional): Past key values for efficient processing.
            inputs_embeds (torch.FloatTensor, optional): Input embeddings.
            labels (torch.LongTensor, optional): Labels for computing the language modeling loss.
            use_cache (bool, optional): Whether to use the model's cache mechanism.
            output_attentions (bool, optional): Whether to output attention weights.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.
        Returns:
            Union[Tuple, AriaCausalLMOutputWithPast]: Model outputs.
        """

        data = {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'pixel_mask': pixel_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'input_embeds': inputs_embeds,
            'labels': labels,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
            'cache_position': cache_position,
            'num_logits_to_keep': num_logits_to_keep,
        }
        visual_outputs = self.vision_tower(
            data['pixel_values'].to(self.visual_encoder.dtype),
            output_hidden_states=True)
        selected_feats = visual_outputs.hidden_states[self.config.visual_select_layer][:, 1:]


        if self.vpt_encoder is None:
            pixel_values = self.projector(selected_feats)
            data['pixel_values'] = pixel_values
        else:
            vpt_regions = data.get('visual_prompts', None)
            visual_feats, visual_prompts = self.prepare_visual_feats(
                selected_feats, 
                vpt_regions, 
            )
            if visual_prompts is not None:
                data.update(visual_prompts) 
            pixel_values = visual_feats

        # prepare data for train/predict
        if self.vpt_encoder is not None:
            data['token_masks'] = self.prepare_token_masks(data['input_ids'])
            
        data = prepare_inputs_labels_for_multimodal(llm=self.language_model, **data)
        # if self.cutoff_len is not None:
        #     for key, value in data.items():
        #         if value is None: continue
        #         if value.shape[1] > self.cutoff_len:
        #             data[key] = value[:, :self.cutoff_len]
            

        outputs = self.language_model(attention_mask=data['attention_mask'],
            position_ids=data['position_ids'],
            past_key_values=data['past_key_values'],
            inputs_embeds=data['inputs_embeds'],
            use_cache=data['use_cache'],
            output_attentions=data['output_attentions'],
            output_hidden_states=data['output_hidden_states'],
            return_dict=data['return_dict'],
            cache_position=data['cache_position'],
            num_logits_to_keep=data['num_logits_to_keep'],
        )
        
        logits = outputs[0]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if data['attention_mask'] is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = data['attention_mask'][:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = data['labels'][..., 1:][
                    shift_attention_mask.to(data['labels'].device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = data['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VTPlugCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     inputs_embeds=None,
    #     pixel_values=None,
    #     attention_mask=None,
    #     cache_position=None,
    #     num_logits_to_keep=None,
    #     **kwargs,
    # ):
    #     # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

    #     model_inputs = self.language_model.prepare_inputs_for_generation(
    #         input_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         cache_position=cache_position,
    #         num_logits_to_keep=num_logits_to_keep,
    #         **kwargs,
    #     )

    #     if cache_position[0] == 0:
    #         # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
    #         # Otherwise we need pixel values to be passed to model
    #         model_inputs["pixel_values"] = pixel_values

    #     return model_inputs

    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        data = {
            'input_ids': inputs,
            'pixel_values': pixel_values,
            'pixel_mask': pixel_mask,
        }
        generation_config = generation_config if generation_config is not None else self.gen_config

        visual_outputs = self.vision_tower(
            data['pixel_values'].to(self.visual_encoder.dtype),
            output_hidden_states=True)
        selected_feats = visual_outputs.hidden_states[self.config.visual_select_layer][:, 1:]


        if self.vpt_encoder is None:
            pixel_values = self.projector(selected_feats)
            data['pixel_values'] = pixel_values
        else:
            vpt_regions = data.get('visual_prompts', None)
            visual_feats, visual_prompts = self.prepare_visual_feats(
                selected_feats, 
                vpt_regions, 
            )
            if visual_prompts is not None:
                data.update(visual_prompts) 
            pixel_values = visual_feats

        # prepare data for train/predict
        if self.vpt_encoder is not None:
            data['token_masks'] = self.prepare_token_masks(data['input_ids'])
            
        data = prepare_inputs_labels_for_multimodal(llm=self.language_model, **data)


        return self.language_model.generate(
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **data,
            **kwargs,
        )

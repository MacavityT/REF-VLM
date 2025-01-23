# Copyright 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import logging
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
import os
from .modules import *



# from .vision_encoder import AriaVisionConfig

logger = logging.getLogger(__name__)

DECODER_CONFIG_CLASS = {
    'box': BoxDecoderConfig,
    'mask': MaskDecoderConfig,
    'pose': PoseDecoderConfig,
    'depth': DepthDecoderConfig
}



# adapted from transformers.models.llava.configuration_llava.LlavaConfig
class VTPlugConfig(PretrainedConfig):
    """
    Configuration class for VT-PLUG model.
    This class handles the configuration for both vision and text components of the VT-PLUG model,
    as well as additional parameters for image token handling and projector mapping.
    Args:
        vision_config (AriaVisionConfig or dict): Configuration for the vision component.
        text_config (AriaMoELMConfig or dict): Configuration for the text component.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query dimensions.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        **kwargs: Additional keyword arguments passed to the parent class.
    Attributes:
        model_type (str): Type of the model, set to "vt_plug".
        is_composition (bool): Whether the model is a composition of multiple components.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query dimensions.
        vision_config (VTPlugConfig): Configuration for the vision component.
        text_config (LLMComfig): Configuration for the text component.
    """

    model_type = "vt_plug"
    is_composition = False

    def __init__(
        self,
        llm='/code/VT-PLUG/checkpoints/Qwen2.5/hf_output/0107_iter105500',
        vision_tower=None,
        mm_projector=None,
        vpt_encoder=None,
        visual_decoder=None,
        ignore_index=-100,
        visual_select_layer=-2,
        image_token_index=151675,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.visual_select_layer = visual_select_layer
        self.image_token_index = image_token_index

        if llm is not None and isinstance(llm, str):
            self.llm_path = llm
            self.text_config = AutoConfig.from_pretrained(llm)
        else:
            raise 'Enter a valid pretrained llm path!'

        if vision_tower is not None and isinstance(vision_tower, str):
            self.vision_config = AutoConfig.from_pretrained(vision_tower)
            self.vision_tower = vision_tower

            attn_implementation = kwargs.pop("attn_implementation", None)
            if attn_implementation is None:
                vision_attn_implementation = "flash_attention_2"
            elif attn_implementation == "sdpa":
                logger.warning(
                    "SDPA is not supported for vit, using flash_attention_2 instead"
                )
                vision_attn_implementation = "flash_attention_2"
            else:
                vision_attn_implementation = attn_implementation
            self.vision_config._attn_implementation = vision_attn_implementation
        # else:
        #     raise 'Enter a valid pretrained visual encoder path!'


        if mm_projector is not None and isinstance(mm_projector, str):
            self.mm_projector_config = AutoConfig.from_pretrained(mm_projector)
            self.mm_projector_path = mm_projector
        elif isinstance(mm_projector, dict):
            self.mm_projector_config = ProjectorConfig(**mm_projector)
            self.mm_projector_path = None
        else:
            self.mm_projector_path = os.path.join(llm,'projector')
            if os.path.exists(self.mm_projector_path):
                self.mm_projector_config = AutoConfig.from_pretrained(self.mm_projector_path)
            else:
                raise 'Enter a valid projector path!'
        
        self.vpt_encoder_path = None
        if vpt_encoder is not None and isinstance(vpt_encoder, str):
            self.vpt_encoder_config = AutoConfig.from_pretrained(vpt_encoder)
            self.vpt_encoder_path = vpt_encoder
        elif isinstance(vpt_encoder, dict):
            self.vpt_encoder_config = VPTEncoderConfig(**vpt_encoder)
        else:
            self.vpt_encoder_path = os.path.join(llm,'vpt_encoder')
            if os.path.exists(self.vpt_encoder_path):
                self.vpt_encoder_config = AutoConfig.from_pretrained(self.vpt_encoder_path)
            else:
                self.vpt_encoder_config = None

        self.decoder_config = {}
        self.decoder_path = {}
        if visual_decoder is not None and isinstance(visual_decoder,dict):
            for key, value in visual_decoder.items():
                if isinstance(value,str):
                    self.decoder_config[key] = AutoConfig.from_pretrained(value)
                    self.decoder_path[key] = value
                elif isinstance(value,dict):
                    self.decoder_config[key] = DECODER_CONFIG_CLASS[key](**value)

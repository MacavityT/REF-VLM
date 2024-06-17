import os
import math
import argparse
from PIL import Image
import numpy as np
import torch
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from mmengine.config import Config, DictAction
from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.dataset.utils import masks_expand2square,mask_transform
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)
from xtuner.utils.constants import VISUAL_PROMPT_PLACEHOLDER,IMAGE_PLACEHOLDER,VISUAL_PROMPT_INDEX
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path





class OkapiInference:

    def __init__(self,
                 config=None,
                 llm_path=None,
                 visual_encoder_path=None,
                 projector_path=None,
                 adapter_path=None,
                 config_options=None,
                 torch_dtype='fp32',
                 bits=None,
                 max_new_tokens=2048,
                 stop_words=[],
                 visual_select_layer=-2):
        
        self.config = config
        self.TORCH_DTYPE_MAP = dict(
            fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

        self.load_model(llm_path=llm_path,
                        visual_encoder_path=visual_encoder_path,
                        projector_path=projector_path,
                        adapter_path=adapter_path,
                        config_options=config_options,
                        torch_dtype=torch_dtype,
                        bits=bits)

        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        self.stop_criteria = get_stop_criteria(
                tokenizer=self.tokenizer, stop_words=self.stop_words)

        self.visual_select_layer = visual_select_layer
    
    def load_model(self,
                llm_path=None,
                visual_encoder_path=None,
                projector_path=None,
                vpt_encoder_path=None,
                adapter_path=None,
                config_options=None,
                torch_dtype='fp32',
                bits=None):

        assert (self.config is not None) or \
            (llm_path and visual_encoder_path and projector_path is not None),\
                "Please input config or model path!"

        if self.config is not None:  # directly load model from config file
            if not os.path.isfile(self.config):
                try:
                    self.config = cfgs_name_path[self.config]
                except KeyError:
                    raise FileNotFoundError(f'Config arg is not None but cannot find {self.config}')
            cfg = Config.fromfile(self.config)
            if config_options is not None:
                cfg.merge_from_dict(config_options)
            model_name = cfg.model.type if isinstance(cfg.model.type,
                                                str) else cfg.model.type.__name__
            self.model = BUILDER.build(cfg.model)

            if cfg.model.get('llm'):
                self.llm = self.model.llm
                self.tokenizer = self.model.tokenizer
                print(f'Load LLM directly from {model_name}')
            else:
                raise f"llm does not exisit in the {model_name}, please edit the config file!"

            if cfg.model.get('visual_encoder'):
                self.visual_encoder = self.model.visual_encoder
                self.image_processor = BUILDER.build(cfg.clip_patch14_336['image_processor'])
                print(f'Load visual encoder directly from {model_name}')
            else:
                raise f"visual encoder does not exisit in the {model_name}, please edit the config file!"

            if cfg.model.get('projector'):
                self.projector = self.model.projector       
                print(f'Load projector from {model_name}')
            else:
                raise f"projector does not exisit in the {model_name}, please edit the config file!"
            
            if cfg.model.get('vpt_encoder'):
                self.vpt_encoder = self.model.vpt_encoder       
                print(f'Load vpt_encoder from {model_name}')
            else:
                raise f"vpt_encoder does not exisit in the {model_name}, please edit the config file!"
        
        else:    # load model from file path
            # build llm
            assert os.path.isdir(llm_path), f"Please verify valid llm path, got {llm_path}"
            assert os.path.isdir(visual_encoder_path), f"Please verify valid visualencoder path, got {visual_encoder_path}"
            assert os.path.isdir(projector_path), f"Please verify valid projector path, got {projector_path}"
            assert os.path.isdir(vpt_encoder_path), f"Please verify valid vpt encoder path, got {vpt_encoder_path}"

            quantization_config = None
            if bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_in_8bit=False,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4')
            elif bits == 8:
                load_in_8bit = True
            llm_model_kwargs = {
                'quantization_config': quantization_config,
                'load_in_8bit': load_in_8bit,
                'device_map': 'auto',
                'trust_remote_code': True,
                'torch_dtype': self.TORCH_DTYPE_MAP[torch_dtype]
            }
            self.llm = AutoModelForCausalLM.from_pretrained(llm_path,**llm_model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                    llm_path,
                    trust_remote_code=True,
                    encode_special_tokens=True)
            print(f'Load LLM from {llm_path}')
            if adapter_path is not None:
                self.llm = PeftModel.from_pretrained(
                    self.llm,
                    adapter_path,
                    trust_remote_code=True)
                print(f'Load adapter from {adapter_path}')

            self.visual_encoder = CLIPVisionModel.from_pretrained(
                visual_encoder_path,
                torch_dtype=self.TORCH_DTYPE_MAP[torch_dtype])
            self.image_processor = CLIPImageProcessor.from_pretrained(
                visual_encoder_path)
            print(f'Load visual_encoder from {visual_encoder_path}')

            self.projector = AutoModel.from_pretrained(
                projector_path,
                torch_dtype=self.TORCH_DTYPE_MAP[torch_dtype],
                trust_remote_code=True)
            print(f'Load projector from {projector_path}')   

            self.vpt_encoder = AutoModel.from_pretrained(
                vpt_encoder_path,
                torch_dtype=self.TORCH_DTYPE_MAP[torch_dtype],
                trust_remote_code=True)
            print(f'Load vpt_encoder from {vpt_encoder_path}')

        self.vpt_encoder.cuda()
        self.vpt_encoder.eval()
        self.projector.cuda()
        self.projector.eval()
        self.visual_encoder.cuda()
        self.visual_encoder.eval()
        self.llm.cuda()
        self.llm.eval()

    def process_image(self,image):
        """Load image and send it to the visual encoder and projector."""
        if isinstance(image,dict):
            image = image['image']
        image = Image.fromarray(image)
        image = expand2square(
            image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0).to(self.visual_encoder.dtype)
        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        selected_feats = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
        pixel_values = self.projector(selected_feats)
        return pixel_values, selected_feats

    def process_vpt(self,selected_feats,prompt_image):
        if prompt_image != []:
            prompt_image = masks_expand2square(prompt_image)

            transformed_masks = []
            for mask in prompt_image:
                transformed_masks.append(
                        mask_transform(mask, self.image_processor)
                    )
            prompt_image = transformed_masks
            visual_prompts = self.vpt_encoder(
                        selected_feats,
                        regions = [prompt_image], 
                        return_dict = True
                    )

        else:

            print("no vpt exists!")
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
                return_dict = True,
            )
            visual_prompts['vpt_count'] = vpt_count

        return visual_prompts, prompt_image        

    def inference(self,
                 prompt_text,
                 n_turn,
                 gen_config,
                 history=None,
                 image=None,
                 prompt_image=None):


        if history is None:
            history = ''
        sep = ''

        history += prompt_text
        print(f"history = {history}")
        pixel_values = None
        selected_feats = None

        if image is not None:
            pixel_values,selected_feats = self.process_image(image)
            visual_prompts,prompt_image = self.process_vpt(selected_feats,
                                                          prompt_image)
            
        if pixel_values is None:
            if n_turn == 0:
                ids = self.tokenizer.encode(history, return_tensors='pt')
            else:
                ids = self.tokenizer.encode(
                    history, return_tensors='pt', add_special_tokens=False)
                
            generate_output = self.llm.generate(
                inputs=ids.cuda(),
                generation_config=gen_config,
                stopping_criteria=self.stop_criteria)

            output_decode = self.tokenizer.decode(generate_output[0])
            history += output_decode

        else:
            chunk_encode = []
            for idx, chunk_image in enumerate(history.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0 and n_turn == 0:
                    add_special_tokens = True
                else:
                    add_special_tokens = False
                if VISUAL_PROMPT_PLACEHOLDER in chunk_image:
                    chunk_vpt = [
                            self.tokenizer.encode(chunk, add_special_tokens=add_special_tokens)
                            for chunk in chunk_image.split(VISUAL_PROMPT_PLACEHOLDER)
                    ]
                else:
                    chunk_vpt = self.tokenizer.encode(chunk_image, add_special_tokens=add_special_tokens)                        
                chunk_encode.append(chunk_vpt)
                
            assert len(chunk_encode) == 2
            ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                if isinstance(cur_chunk_encode[0], list):
                    for idx_vpt, cur_chunk_encode_vpt in enumerate(cur_chunk_encode):
                        ids.extend(cur_chunk_encode_vpt)
                        if idx_vpt != len(cur_chunk_encode) - 1:
                            ids.append(VISUAL_PROMPT_INDEX)
                    
                else:
                    ids.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        ids.append(IMAGE_TOKEN_INDEX)            
            ids = torch.tensor(ids).cuda().unsqueeze(0)
            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=self.llm, input_ids=ids, pixel_values=pixel_values,**visual_prompts)
            for key in mm_inputs.keys():
                if mm_inputs[key] is not None:
                    mm_inputs[key] = mm_inputs[key].to(self.llm.dtype)

            generate_output = self.llm.generate(
                **mm_inputs,
                generation_config=gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria)
            output_decode = self.tokenizer.decode(generate_output[0])
            history += output_decode

        n_turn +=1
        history += sep
        if len(self.tokenizer.encode(history)) >= self.max_new_tokens:
            print(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation {self.max_new_tokens}.')
            n_turn = 0
            history = ''
            prompt_image = []

        return {
            'output': output_decode,
            'history': history,
            'prompt_image': prompt_image,
            'n_turn': n_turn
        }
    

    def __call__(self,
                 prompt_text,
                 n_turn,
                 history,
                 image,
                 prompt_image,
                 temperature,
                 top_p,
                 top_k,
                 repetition_penalty=1.0,
                 ):
        """forward function to make the inference"""

        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )

        output_dict = self.inference(
            prompt_text=prompt_text,
            n_turn=n_turn,
            history=history,
            image=image,
            prompt_image=prompt_image,
            gen_config=gen_config
        )

        return output_dict['output'], output_dict['history'], output_dict['prompt_image'], output_dict['n_turn']
    


import os
import torch
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from mmengine.config import Config, DictAction
from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path




def process_image(image_path,
                  image_processor,
                  visual_encoder,
                  projector,
                  visual_select_layer=-2):
    """Load image from image path and send it to the visual encoder and projector."""
    if image_path == '':
        return None
    image = load_image(image_path)
    image = expand2square(
        image, tuple(int(x * 255) for x in image_processor.image_mean))
    image = image_processor.preprocess(
        image, return_tensors='pt')['pixel_values'][0]
    image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)
    visual_outputs = visual_encoder(image, output_hidden_states=True)
    pixel_values = projector(
        visual_outputs.hidden_states[visual_select_layer][:, 1:])
    return pixel_values



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
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 repetition_penalty=1.0,
                 stop_words=[]):
        
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
        self.gen_config = GenerationConfig(
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
    
    def load_model(self,
                llm_path=None,
                visual_encoder_path=None,
                projector_path=None,
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
                self.image_processor = BUILDER.build(cfg.okapi_dataset['image_processor'])
                print(f'Load visual encoder directly from {model_name}')
            else:
                raise f"visual encoder does not exisit in the {model_name}, please edit the config file!"

            if cfg.model.get('projector'):
                self.projector = self.model.projector       
                print(f'Load projector from {model_name}')
            else:
                raise f"projector does not exisit in the {model_name}, please edit the config file!"
        
        else:    # load model from file path
            # build llm
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

        self.projector.cuda()
        self.projector.eval()
        self.visual_encoder.cuda()
        self.visual_encoder.eval()
        self.llm.cuda()
        self.llm.eval()

    def inference(self,
                 text,
                 history=None,
                 image=None,
                 n_turn=0,
                 bot_name='okapi',
                 prompt_template='vicuna',
                 system_template=None,
                 system=None):
        
        if history is None:
            history = ''
        sep = ''
        if prompt_template:
            template = PROMPT_TEMPLATE[prompt_template]
            self.stop_words += template.get('STOP_WORDS', [])
            sep = template.get('SEP', '')
        if prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[prompt_template]
            if 'SYSTEM' in template and n_turn == 0:
                system_text = None
                if system_template is not None:
                    system_text = SYSTEM_TEMPLATE[
                        system_template].format(
                            round=n_turn + 1, bot_name=bot_name)
                elif system is not None:
                    system_text = system
                elif system_template is not None and system is not None:
                    print("Warning: both system and system_template are not None, use system.")
                if system_text is not None:
                    prompt_text += template['SYSTEM'].format(
                        system=system_text,
                        round=n_turn + 1,
                        bot_name=bot_name)
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=n_turn + 1, bot_name=bot_name)
        else:
            prompt_text = text

        history += prompt_text
        pixel_values = None
        if image is not None:
            pixel_values = process_image(image,self.image_processor,
                                         self.visual_encoder,self.projector)
            
        if pixel_values is None:
            if n_turn == 0:
                ids = self.tokenizer.encode(history, return_tensors='pt')
            else:
                ids = self.tokenizer.encode(
                    history, return_tensors='pt', add_special_tokens=False)
            generate_output = self.llm.generate(
                inputs=ids.cuda(),
                generation_config=self.gen_config,
                stopping_criteria=self.stop_criteria)

            output_decode += self.tokenizer.decode(generate_output[0])
            history += output_decode

        else:
            chunk_encode = []
            for idx, chunk in enumerate(history.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0 and n_turn == 0:
                    cur_encode = self.tokenizer.encode(chunk)
                else:
                    cur_encode = self.tokenizer.encode(
                        chunk, add_special_tokens=False)
                chunk_encode.append(cur_encode)
            assert len(chunk_encode) == 2
            ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    ids.append(IMAGE_TOKEN_INDEX)
            ids = torch.tensor(ids).cuda().unsqueeze(0)
            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=self.llm, input_ids=ids, pixel_values=pixel_values)

            generate_output = self.llm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria)
            output_decode += self.tokenizer.decode(generate_output[0])
            history += output_decode

        n_turn +=1
        history += sep
        if len(history) >= self.max_new_tokens:
            print(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation {self.max_new_tokens}.')
            n_turn = 0
            history = ''

        return {
            'output': output_decode,
            'history': history,
            'n_turn': n_turn
        }
    

    def __call__(self,
                 text,
                 history,
                 image,
                 n_turn,
                 **kwargs):
        """forward function to make the inference"""

        bot_name = kwargs.pop('bot_name','okapi')
        prompt_template = kwargs.pop('prompt_template','vicuna')
        system_template = kwargs.pop('system_template',None)
        system = kwargs.pop('system',None)

        output_dict = self.inference(
            text=text,
            history=history,
            image=image,
            n_turn=n_turn,
            bot_name=bot_name,
            prompt_template=prompt_template,
            system_template=system_template,
            system=system
        )

        return "", output_dict['output'], output_dict['history'], output_dict['n_turn']
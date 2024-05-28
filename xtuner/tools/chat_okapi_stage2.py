# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import sys
import math
import time
import torch
import numpy as np
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers.generation.streamers import TextStreamer
from mmengine.config import Config, DictAction
from xtuner.configs import cfgs_name_path
from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

# import debugpy
# debugpy.connect(('127.0.0.1', 5577))

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

SYS_TEMPLATE_OKAPI = [
    'Task Command:\n- task name: vqa\n- answer element: sentence\n- unit: None',
    'Task Command:\n- task name: detection\n- answer element: phrase, unit\n- unit: <Unit>box</Unit>\n',
    'Task Command:\n- task name: segmentation\n- answer element: phrase, unit\n- unit: <Unit>mask</Unit>\n',
    'Task Command:\n- task name: grounding_detection\n- answer element: phrase, unit\n- unit: <Unit>box</Unit>\n',
    'Task Command:\n- task name: grounding_segmentation\n- answer element: phrase, unit\n- unit: <Unit>mask</Unit>\n',
    'Task Command:\n- task name: gcg_detection\n- answer element: phrase, sentence, unit\n- unit: <Unit>box</Unit>\n',
    'Task Command:\n- task name: gcg_segmentation\n- answer element: phrase, sentence, unit\n- unit: <Unit>mask</Unit>\n',
]


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')



    parser.add_argument(
        '--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--okapi', default=None, help='okapi name or path')
    parser.add_argument(
        '--config', default=None,help='config file name or path.')
    parser.add_argument(
        '--llm', default=None, help='llm path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--vpt-encoder', default=None, help='vpt encoder name or path')
    parser.add_argument(
        '--projector', default=None, help='projector name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument(
        '--load-model-from-pth', default=False, help='directly load pth pretrained model')
    parser.add_argument(
         '--checkpoint', default=None,type=str, help='model checkpoint path')
    parser.add_argument(
        '--torch-dtype',
        default='fp32',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default='okapi',
        help='Specify a prompt template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, '
               'RESET: reset history, IMAGE: add a new image) >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result

def get_image_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\nInput image path, double enter to end input >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result

def get_system_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    valid_list = [1,2,3,4,5,6,7]
    while result is None:
        print(('\n Please input numbers: [1] vqa [2] detection [3] segmentation [4] grounding detection [5] grounding segmentation [6] gcg detection [7] gcg segmentation >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')

        if result.isdigit():
            if int(result) not in valid_list:
                result = None
        else:
            result = None

    return int(result)

def process_image(args,image_path,image_processor,visual_encoder,projector):
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
    selected_feats = visual_outputs.hidden_states[args.visual_select_layer][:, 1:]
    pixel_values = projector(selected_feats)
    return pixel_values, selected_feats

def process_vpt(selected_feats,vpt_encoder,visual_prompts=None):
    if visual_prompts is not None:
        visual_prompts = vpt_encoder(
                    selected_feats,
                    regions = visual_prompts, 
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
        visual_prompts = vpt_encoder(
            selected_feats,
            regions = regions, 
            return_dict = True,
        )
        visual_prompts['vpt_count'] = vpt_count

    return visual_prompts

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # parse config
    if args.config is not None:
        if not osp.isfile(args.config):
            try:
                args.config = cfgs_name_path[args.config]
            except KeyError:
                raise FileNotFoundError(f'Config arg is not None but cannot find {args.config}')
        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }
    if args.lagent:
        from lagent.actions import ActionExecutor, GoogleSearch
        from lagent.agents import (CALL_PROTOCOL_CN, FORCE_STOP_PROMPT_CN,
                                   ReAct, ReActProtocol)
        from lagent.llms import HFTransformerCasualLM

        try:
            SERPER_API_KEY = os.environ['SERPER_API_KEY']
        except Exception:
            print('Please obtain the `SERPER_API_KEY` from https://serper.dev '
                  'and set it using `export SERPER_API_KEY=xxx`.')
            sys.exit(1)

        model_kwargs.pop('trust_remote_code')
        assert args.llm is not None
        llm = HFTransformerCasualLM(
            args.llm, model_kwargs=model_kwargs)
        if args.adapter is not None:
            print(f'Loading adapter from {args.adapter}...')
            llm.model = PeftModel.from_pretrained(
                llm.model,
                args.adapter,
                offload_folder=args.offload_folder,
                trust_remote_code=True)
        search_tool = GoogleSearch(api_key=SERPER_API_KEY)
        chatbot = ReAct(
            llm=llm,
            action_executor=ActionExecutor(actions=[search_tool]),
            protocol=ReActProtocol(
                call_protocol=CALL_PROTOCOL_CN,
                force_stop=FORCE_STOP_PROMPT_CN))
        while True:
            text = get_input()
            while text.strip() == 'RESET':
                print('Log: History responses have been removed!')
                chatbot._session_history = []
                inputs = ''
                text = get_input()
            if text.strip() == 'EXIT':
                print('Log: Exit!')
                exit(0)
            response = chatbot.chat(text)
            print(response.response)
    else:        
        # load model
        if args.config is not None:
            model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
            model = BUILDER.build(cfg.model)

            if cfg.model.get('llm') and cfg.model.get('freeze_llm', True):
                llm = model.llm
                tokenizer = model.tokenizer
                print(f'Load LLM directly from {model_name}')

            if cfg.model.get('visual_encoder') and cfg.model.get('freeze_visual_encoder', True):
                visual_encoder = model.visual_encoder
                image_processor = BUILDER.build(cfg.train_dataset['image_processor'])
                print(f'Load visual encoder directly from {model_name}')

        if args.load_model_from_pth:
            assert args.checkpoint is not None, "Please add valid checkpoint path!"
            state_dict = guess_load_checkpoint(args.checkpoint)
            model.load_state_dict(state_dict, strict=False)
            llm = model.llm
            tokenizer = model.tokenizer
            projector = model.projector
            visual_encoder = model.visual_encoder
            vpt_encoder = model.vpt_encoder
            image_processor = BUILDER.build(cfg.train_dataset['image_processor'])
            print("Load pretrained checkpoint pth success.")

        else:
            # build visual_encoder & projector directly from the path
            if args.okapi is not None:
                okapi_path = snapshot_download(
                    repo_id=args.okapi) if not osp.isdir(
                        args.okapi) else args.okapi
                
                llm = AutoModelForCausalLM.from_pretrained(okapi_path,**model_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(
                    okapi_path,
                    trust_remote_code=True,
                    encode_special_tokens=True)
                print(f'Load LLM from {args.llm}')
                
                if 'visual_encoder' in os.listdir(okapi_path):
                    visual_encoder_path = osp.join(okapi_path, 'visual_encoder')
                    visual_encoder = CLIPVisionModel.from_pretrained(
                            visual_encoder_path,
                            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
                    image_processor = CLIPImageProcessor.from_pretrained(
                            visual_encoder_path)
                    print(f'Load visual_encoder from {visual_encoder_path}')
                
                if 'projector' in os.listdir(okapi_path):
                    projector_path = osp.join(okapi_path, 'projector')
                    projector = AutoModel.from_pretrained(
                        projector_path,
                        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                        trust_remote_code=True)
                    print(f'Load projector from {projector_path}')

                if 'vpt_encoder' in os.listdir(okapi_path):
                    vpt_encoder_path = osp.join(okapi_path, 'vpt_encoder')
                    vpt_encoder = AutoModel.from_pretrained(
                        vpt_encoder_path,
                        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                        trust_remote_code=True)
                    print(f'Load vpt encoder from {vpt_encoder_path}')
            else:
                # build model from seperate path
                if args.llm is not None:
                    llm = AutoModelForCausalLM.from_pretrained(args.llm,**model_kwargs)
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.llm,
                        trust_remote_code=True,
                        encode_special_tokens=True)
                    print(f'Load LLM from {args.llm}')
                if args.adapter is not None:
                    llm = PeftModel.from_pretrained(
                        llm,
                        args.adapter,
                        offload_folder=args.offload_folder,
                        trust_remote_code=True)
                    print(f'Load adapter from {args.adapter}')
                if args.visual_encoder is not None:
                    visual_encoder = CLIPVisionModel.from_pretrained(
                        args.visual_encoder,
                        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
                    image_processor = CLIPImageProcessor.from_pretrained(
                        args.visual_encoder)
                    print(f'Load visual_encoder from {args.visual_encoder}')
                if args.projector is not None:
                    projector = AutoModel.from_pretrained(
                        args.projector,
                        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                        trust_remote_code=True)
                    print(f'Load projector from {args.projector}')
                if args.vpt_encoder is not None:
                    vpt_encoder = AutoModel.from_pretrained(
                        args.vpt_encoder_path,
                        torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
                        trust_remote_code=True)
                    print(f'Load vpt encoder from {args.vpt_encoder_path}')

        vpt_encoder.cuda()
        vpt_encoder.eval()
        projector.cuda()
        projector.eval()
        visual_encoder.cuda()
        visual_encoder.eval()
        llm.cuda()
        llm.eval()

        stop_words = args.stop_words
        sep = ''
        if args.prompt_template:
            template = PROMPT_TEMPLATE[args.prompt_template]
            stop_words += template.get('STOP_WORDS', [])
            sep = template.get('SEP', '')
        stop_criteria = get_stop_criteria(
            tokenizer=tokenizer, stop_words=stop_words)
        if args.no_streamer:
            streamer = None
        else:
            streamer = TextStreamer(tokenizer, skip_prompt=True)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

        n_turn = 0
        inputs = ''
        pixel_values = None
        while True:
            system_id = get_system_input()
            if n_turn == 0 :
                image_path = get_image_input()
                if os.path.isfile(image_path):
                    pixel_values, selected_feats = process_image(args,image_path,image_processor,visual_encoder,projector)
                else:
                    print(f'Warning: image path [{image_path}] is None or not a valid file! Ignore the image path.')
                
            text = get_input()
            while text.strip() == 'RESET':
                print('Log: History responses have been removed!')
                n_turn = 0
                inputs = ''
                pixel_values = None
                system_text = get_system_input()
                image_path = get_image_input()
                if os.path.isfile(image_path):
                    pixel_values, selected_feats = process_image(args,image_path,image_processor,visual_encoder,projector)
                else:
                    print(f'Warning: image path [{image_path}] is None or not a valid file! Ignore the image path.')
                text = get_input()
            while text.strip() == 'IMAGE':
                print('Log: Please input a new image path!')
                image_path = get_image_input()
                if os.path.isfile(image_path):
                    pixel_values, selected_feats = process_image(args,image_path,image_processor,visual_encoder,projector)
                else:
                    print(f'Warning: image path [{image_path}] is None or not a valid file! Ignore the image path.')
                text = get_input()
            if text.strip() == 'EXIT':
                print('Log: Exit!')
                exit(0)

            system_task_text = SYS_TEMPLATE_OKAPI[system_id-1]
            if pixel_values is not None and n_turn == 0:
                text = DEFAULT_IMAGE_TOKEN + '\n' + text

            if args.prompt_template:
                prompt_text = ''
                template = PROMPT_TEMPLATE[args.prompt_template]
                system_text = None
                if 'SYSTEM' in template:
                    if n_turn == 0:
                        system_text = template['SYSTEM_PREFIX'] + template['SYSTEM'].format(system=system_task_text,
                                                                                            round=n_turn + 1,
                                                                                            bot_name=args.bot_name)
                    else:
                        system_text = template['SYSTEM'].format(system=system_task_text,
                                                                round=n_turn + 1,
                                                                bot_name=args.bot_name)
                    if system_text is not None:
                        prompt_text += system_text
                prompt_text += template['INSTRUCTION'].format(
                    input=text, round=n_turn + 1, bot_name=args.bot_name)
                
            else:
                prompt_text = text
            inputs += prompt_text
            if pixel_values is None:
                if n_turn == 0:
                    ids = tokenizer.encode(inputs, return_tensors='pt')
                else:
                    ids = tokenizer.encode(
                        inputs, return_tensors='pt', add_special_tokens=False)

                time1 = time.time()
                generate_output = llm.generate(
                    inputs=ids.cuda(),
                    generation_config=gen_config,
                    streamer=streamer,
                    stopping_criteria=stop_criteria)
                time2 = time.time()
                if streamer is None:
                    output_text = tokenizer.decode(
                        generate_output[0][len(ids[0]):])
                    end = '' if output_text[-1] == '\n' else '\n'
                    print(output_text, end=end)
                    print(f"total inference time:{time2-time1}")
                    print(f"token length = {len(generate_output[0][len(ids[0]):])}")
                inputs += tokenizer.decode(generate_output[0])

            else:
                chunk_encode = []
                for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                    if idx == 0 and n_turn == 0:
                        cur_encode = tokenizer.encode(chunk)
                    else:
                        cur_encode = tokenizer.encode(
                            chunk, add_special_tokens=False)
                    chunk_encode.append(cur_encode)
                assert len(chunk_encode) == 2
                ids = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    ids.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        ids.append(IMAGE_TOKEN_INDEX)
                ids = torch.tensor(ids).cuda().unsqueeze(0)
                visual_prompts = process_vpt(selected_feats,vpt_encoder)
                mm_inputs = prepare_inputs_labels_for_multimodal(
                    llm=llm, input_ids=ids, pixel_values=pixel_values,**visual_prompts)

                generate_output = llm.generate(
                    **mm_inputs,
                    generation_config=gen_config,
                    streamer=streamer,
                    bos_token_id=tokenizer.bos_token_id,
                    stopping_criteria=stop_criteria)
                if streamer is None:
                    output_text = tokenizer.decode(generate_output[0])
                    end = '' if output_text[-1] == '\n' else '\n'
                    print(output_text, end=end)
                inputs += tokenizer.decode(generate_output[0])

            n_turn += 1
            inputs += sep
            if len(inputs) >= args.max_new_tokens:
                print(
                    'Remove the memory of history responses, since '
                    f'it exceeds the length limitation {args.max_new_tokens}.')
                n_turn = 0
                inputs = ''


if __name__ == '__main__':
    main()

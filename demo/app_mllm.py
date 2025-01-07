from typing import List,Union
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import cv2
import re
import os
import copy
import time
import random
from PIL import Image

from vt_plug.utils import PROMPT_TEMPLATE,VISUAL_PROMPT_PLACEHOLDER,BOV_TOKEN,EOV_TOKEN,VISUAL_REPRESENTATION_TOKEN
from vt_plug.utils.constants import MASKS_PLACEHOLDER
from vt_plug.dataset import VTInstructDataset
from vt_plug.dataset.collate_fns import vt_collate_fn
from vt_plug.dataset.map_fns.dataset_map_fns.vt_map_fn_stage2 import get_cot_elements
from inference import VTPlugInference
from utils import SingleInferDataset
from mmengine.config import Config, DictAction
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path

STOP_WORDS = ['<|im_end|>', '<|endoftext|>','</s>']

def denormalize_image(image):
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
    denormalized_tensor = image.cpu() * std + mean
    denormalized_tensor = denormalized_tensor[0].permute(1,2,0).detach().cpu().numpy()
    output_denormalized_image = np.clip(denormalized_tensor * 255,0,255).astype(np.uint8)
    return output_denormalized_image



def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        '--config', help='config file name or path.')
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


def post_process_generate_ids(tokenizer, ids: torch.Tensor):
    ids = copy.deepcopy(ids)  # do not modify origin preds and targets
    if ids[ids < 0].cpu().numpy().tolist() != []:
        ids[ids < 0] = tokenizer.pad_token_id
    return ids

def decode_generate_ids(tokenizer, ids: torch.Tensor,skip_special_tokens=True) -> Union[List[str], str]:
    assert ids.ndim in [1, 2]
    only_one_sentence = ids.ndim == 1
    if only_one_sentence:
        ids = ids.unsqueeze(0)
    ids = post_process_generate_ids(tokenizer,ids)
    res = tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
    if only_one_sentence:
        return res[0]
    return res

class ImageSketcher(gr.Image):
    """
    Code is from https://github.com/jshilong/GPT4RoI/blob/7c157b5f33914f21cfbc804fb301d3ce06324193/gpt4roi/app.py#L365

    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'. 
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == 'boxes' and self.source in ['upload', 'webcam']:
            if isinstance(x, str):
                x = {'image': x, 'boxes': []}
            else:
                assert isinstance(x, dict)
                assert isinstance(x['image'], str)
                assert isinstance(x['boxes'], list)

        x = super().preprocess(x)
        return x

def init_image(img):
    if isinstance(img, dict):
        img = img["image"]

    if isinstance(img, List):
        img = cv2.imread(img[0])
        img = img[:, :, ::-1]

    h_, w_ = img.shape[:2]
    if w_ > 640:
        ratio = 640 / w_
        new_h, new_w = int(h_ * ratio), int(w_ * ratio)
        preprocessed_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        print(new_h, new_w)
    else:
        preprocessed_img = img.copy()

    return (
        preprocessed_img,
        preprocessed_img,
    )



# import debugpy
# debugpy.connect(('127.0.0.1', 5577))

args = parse_args()
torch.manual_seed(args.seed)

# directly load model from config file
if not os.path.isfile(args.config):
    try:
        config = cfgs_name_path[args.config]
    except KeyError:
        raise FileNotFoundError(f'Config arg is not None but cannot find {args.config}')
else:
    config = args.config
cfg = Config.fromfile(config)
if args.cfg_options is not None:
    cfg.merge_from_dict(args.config_options)

model = BUILDER.build(cfg.model)
tokenizer = model.tokenizer
model.cuda().eval()


def submit_step1(input_system,input_text, input_image, chatbot, state, prompt_image_list):

    system_value = input_system
    input_text_chatbot = input_text
    pattern = r"<masks>"
    input_text_chatbot = re.sub(pattern, r"**\<masks\>**", input_text_chatbot)

    chatbot = chatbot + [[input_text_chatbot, None]]
    cur_prompt_image_list = []


    if input_image is not None:
        if isinstance(input_image,np.ndarray):
            state['input_data'].add_image(input_image)
        elif isinstance(input_image,dict):
            state['input_data'].add_image(input_image['image'])
    state['input_data'].append_message_question(input_text,system_value,cur_prompt_image_list)
    state['input_data'].add_one_conversation()
    state['input_data_okapi'].add_dataset(state['input_data'])
    state['dataloader'] = DataLoader(batch_size=1,num_workers=0,dataset=state['input_data_okapi'],collate_fn=vt_collate_fn)


    return "", chatbot, state, prompt_image_list, input_image


def submit_step2(chatbot, state,prompt_image_list,temperature,top_p,top_k):

    
    if chatbot[-1][0] == "":
        raise gr.Error("Please enter prompt sentences in the chatbox!")
    
    model.gen_config.temperature = temperature
    model.gen_config.top_p = top_p
    model.gen_config.top_k = top_k
    with torch.no_grad():
        for idx, data_batch in enumerate(state['dataloader']):
            for key in data_batch['data'].keys():
                if isinstance(data_batch['data'][key],torch.Tensor):
                    data_batch['data'][key] = data_batch['data'][key].cuda()
            output = model(**data_batch,mode='predict')

    print("prompt length: ", len(prompt_image_list))
    input_str = decode_generate_ids(tokenizer,data_batch['data']['input_ids'],skip_special_tokens=False)
    print(f'input:{input_str}')

    output_seq = decode_generate_ids(tokenizer,output[0]['generate_ids'],skip_special_tokens=False)
    for stop_word in STOP_WORDS:
        output_seq = output_seq.replace(stop_word,'').strip()
    output_seq = output_seq.replace("<","\<").replace(">","\>")


    bot_message = output_seq
    print(output_seq)
    chatbot[-1][1] = ""

    for character in bot_message:
        chatbot[-1][1] += character
        time.sleep(0.005)
        yield chatbot
    
    return chatbot



def clear_states(preprocessed_img,prompt_image_list,chatbot,task_state):
    return None,[],[],{
                        'previous_system':None,
                        'input_data':SingleInferDataset(),
                        'input_data_okapi':BUILDER.build(cfg.infer_dataset),
                        'dataloader': None,
                        'prompt_input':None}

theme = gr.themes.Default()

# title_markdown = ("""
# ![LOGO](/code/okapi-mllm/demo/assets/logo/logo3.png)
# # 🌋 Ladon: Multi-Visual Tasks Multimodal Large Language Model
# [[Project Page]](https://github.com/MacavityT/okapi-mllm/) [[Paper]](https://github.com/MacavityT/okapi-mllm/) [[Code]](https://github.com/MacavityT/okapi-mllm/) [[Model]](https://github.com/MacavityT/okapi-mllm/)
# """)

# title_markdown = ("""
# <p align="center">
#   <a href="#">
# <img src="https://i.mij.rip/2024/06/12/845590e05554cc3b25907dcb0649469a.md.png" alt="Logo" width="130"></a>
#   <h4 align="center"><font color="#966661">Ladon</font>: Multi-Visual Tasks Multimodal Large Language Model</h4>
#   <p align="center">
#     <a href='https://github.com/MacavityT/okapi-mllm/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
#     <a href='https://github.com/MacavityT/okapi-mllm/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
#     <a href='https://github.com/MacavityT/okapi-mllm/'><img src='https://img.shields.io/badge/Online-Demo-green'></a>
#   </p>
# </p>
# """)



# title_markdown = ("""
# <p align="center">
#   <a href="#">
# <img src="https://i2.mjj.rip/2024/06/12/c1f69496f89fee9eca4a5a1c279e373b.png" alt="Logo"></a>
# </p>
# """)

title_markdown = ("""
# 🌋 YongZhi-MLLM
[[Project Page]](https://github.com/MacavityT/VT-PLUG) [[Paper]](https://github.com/MacavityT/VT-PLUG) [[Code]](https://github.com/MacavityT/VT-PLUG) [[Model]](https://github.com/MacavityT/VT-PLUG)
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")


with gr.Blocks(
    title="Ladon: Multi-Visual Tasks Multimodal Large Language Model",
    theme=theme
) as demo:
    
    preprocessed_img = gr.State(value=None)
    prompt_image_list = gr.State(value=[])
    task_state = gr.State(value={'previous_system':None,
                                 'input_data':SingleInferDataset(),
                                 'input_data_okapi':BUILDER.build(cfg.infer_dataset),
                                 'dataloader': None,
                                 'prompt_input':None})
    example_list_image = [
        [os.path.join(os.path.dirname(__file__), "assets/dog.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/fishing.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/rec_bear.png")],
        [os.path.join(os.path.dirname(__file__), "assets/woman.jpeg")],
        [os.path.join(os.path.dirname(__file__), "assets/view.jpg")],
    ]


    example_list = [
         ["Describe the main features of the image."],
         ["Please describe the image in more details."],
         ["Where is the dog?"],
         ["请你描述这张图片所展示的信息。"],
    ]



    descrip_chatbot = """
                    ### 💡 Tips:

                    🧸 Upload an image, and you can pull a frame on the image to select the area of interest.

                    🖱️ Then click the **Generate box and description** button to generate segmentation and description accordingly.

                    🔔 If you want to choose another area or switch to another photo, click the button ↪️ first.

                    ❗️ If there are more than one box, the last one will be chosen.

                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.

                    📌 Click the button **Clear Image** to clear the current image. 
    
    """
    # gr.Markdown(title_markdown)

    with gr.Row():
        # gr.HTML(
        #     """
        #     <p align="center">
        #     <a href="#">
        #     <img src="https://i.mij.rip/2024/06/12/845590e05554cc3b25907dcb0649469a.md.png" alt="Logo" width="200"></a>
        #     </p>
        #     <h1 style="text-align: center; font-weight: 800; font-size: 2rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        #     Ladon: Multi-Visual Tasks Multimodal Large Language Model
        #     """
        # )
        gr.HTML(
            """
            <p align="center">
            <a href="#">
            <img src="https://i2.mjj.rip/2024/06/12/c1f69496f89fee9eca4a5a1c279e373b.png" alt="Logo" width="1000"></a>
            </p>
            """
        )

    with gr.Row():
        gr.Markdown(
            "[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://github.com/MacavityT/okapi-mllm)"
        )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    # input_img = ImageSketcher(type="numpy", label="Input Image", height=550)
                    input_img = gr.Image(type='numpy', label="Input Image", height=550)
                    
                    system_text = gr.Textbox(label="System",placeholder='System Messages')          
                    input_text = gr.Textbox(label="Input Instruction",placeholder='Please Input Messages')
                    submit_button = gr.Button("Submit", variant="primary")
                    example_data = gr.Dataset(
                        label="Image Examples",
                        components=[input_img],
                        samples=example_list_image,
                    )
                    gr.Examples(
                        examples=example_list,
                        inputs=[input_text],
                    )


        with gr.Column():
            with gr.Row():
                temp_slider = gr.Slider(minimum=0.0,maximum=1.0,value=0.1,
                                step=0.05,interactive=True,label="Temperature")
                top_p_slider = gr.Slider(minimum=0.0,maximum=1.0,value=0.75,
                                step=0.05,interactive=True,label="Top P")
                top_k_slider = gr.Slider(minimum=0,maximum=100,value=40,
                                step=1,interactive=True,label="Top K")
                
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(height=800)    
                    clear_button = gr.Button("🗑 Clear Button")
                    gr.Markdown(descrip_chatbot)

    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)


    
    clear_button.click(clear_states,
                         [preprocessed_img,prompt_image_list,chatbot,task_state],
                         [preprocessed_img,prompt_image_list,chatbot,task_state]).then(
        lambda: None,
        None,
        None,
        _js="() => {document.body.innerHTML='<h1 style=\"font-family:monospace;margin-top:20%;color:lightgray;text-align:center;\">Reloading...</h1>'; setTimeout(function(){location.reload()},2000); return []}",
    )
    
    
    input_img.upload(
        init_image,
        [input_img],
        [
            preprocessed_img,
            input_img
        ],
    )


    submit_button.click(submit_step1,
                        [system_text,input_text,input_img,chatbot,task_state,prompt_image_list],
                        [input_text,chatbot,task_state,prompt_image_list]).then(
                        submit_step2,
                        [chatbot,task_state,prompt_image_list,temp_slider,top_p_slider,top_k_slider],
                        [chatbot]
                        )
    


    example_data.click(clear_states,
                         [preprocessed_img,prompt_image_list,chatbot,task_state],
                         [preprocessed_img,prompt_image_list,chatbot,task_state]).then(
        init_image,
        [example_data],
        [
            preprocessed_img,
            input_img,
        ],
    )


demo.queue().launch(
    debug=True,server_port=6295,
)
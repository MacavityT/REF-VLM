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

from xtuner.utils import PROMPT_TEMPLATE,DEFAULT_IMAGE_TOKEN,VISUAL_PROMPT_PLACEHOLDER,BOV_TOKEN,EOV_TOKEN,VISUAL_REPRESENTATION_TOKEN
from xtuner.utils.constants import MASKS_PLACEHOLDER
from xtuner.dataset import OkapiDataset
from xtuner.dataset.collate_fns import okapi_collate_fn
from xtuner.dataset.map_fns.dataset_map_fns.okapi_map_fn_stage2 import get_cot_elements
from xtuner.dataset.utils import (visualize_box,
                                  visualize_mask,
                                  mask_square2origin,
                                  draw_label_type,
                                  denorm_box_xywh_square2origin)
from inference import OkapiInference
from utils import SingleInferDataset
from mmengine.config import Config, DictAction
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path





def choose_system(task_name):
    if task_name == 'VQA':
        return {'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}
    elif task_name == 'Detection':
        return {'task':{'task_name':'detection','element':['phrase'],'use_unit':True},'unit':['box']}
    elif task_name == 'Segmentation':
        return {'task':{'task_name':'segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}
    elif task_name == 'Grounding Detection':
        return {'task':{'task_name':'grounding_detection','element':['phrase'],'use_unit':True},'unit':['box']}
    elif task_name == 'Grounding Segmentation':
        return {'task':{'task_name':'grounding_segmentation','element':['phrase'],'use_unit':True},'unit':['mask']}
    elif task_name == 'Gcg Detection':
        return {'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']}
    elif task_name == 'Gcg Segmentation':
        return {'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']}
    else:
        raise NotImplementedError


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
        preprocessed_img,
        preprocessed_img,
        preprocessed_img,
        [],
    )


def img_select_point(original_img: np.ndarray, sel_pix: list, evt: gr.SelectData, point_mask: dict):
    img = original_img.copy()
    sel_pix.clear()
    sel_pix.append((evt.index, 1))  # append the foreground_point
    point_mask_img = np.zeros((img.shape[0],img.shape[1]))
    # draw points
    for point, label in sel_pix:
        cv2.circle(img, point, 10, (240, 240, 240), -1, 0)
        cv2.circle(img, point, 10, (30, 144, 255), 2, 0)
        cv2.circle(point_mask_img, point, 10, (255, 255, 255), -1, 0)
    sel_pix.clear()
    point_mask['point'] = (point_mask_img/255.).astype(np.uint8)
    return img, point_mask


def img_select_box(input_image: dict, prompt_image_list: list):
    if 'boxes' in input_image.keys():
        box_mask = np.zeros((input_image['image'].shape[0],input_image['image'].shape[1]))
        boxes = input_image['boxes'][-1]
        pt1 = (int(boxes[0]),int(boxes[1]))
        pt2 = (int(boxes[2]),int(boxes[3]))
        cv2.rectangle(box_mask, pt1, pt2, (255, 255, 255), -1)
        prompt_image_list.append((box_mask/255.).astype(np.uint8))
    return prompt_image_list


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


def submit_step1(input_text, input_image, chatbot, radio, state, prompt_image_list, point_mask=None):
    if radio == "":
        return "", chatbot, state, prompt_image_list,radio

    system_value = choose_system(radio)
    input_text_chatbot = input_text
    pattern = r"<masks>"
    input_text_chatbot = re.sub(pattern, r"**\<masks\>**", input_text_chatbot)

    chatbot = chatbot + [[input_text_chatbot, None]]
    cur_prompt_image_list = []

    if isinstance(input_image,dict):
        if 'boxes' in input_image.keys():
            if input_text.count(MASKS_PLACEHOLDER) == 1:
                if len(input_image['boxes']) > 1:
                    gr.Warning("Multiple boxes will appear in the image; only the last box plotted will be selected.")
                cur_prompt_image_list = img_select_box(input_image,cur_prompt_image_list)
            else:
                gr.Warning(f"Number of {MASKS_PLACEHOLDER} is {input_text.count(MASKS_PLACEHOLDER)} \
                           not equal to 1, we do not record the box that you plot!")
        elif 'mask' in input_image.keys():
            if input_text.count(MASKS_PLACEHOLDER) == 1:
                mask = (input_image['mask'][:,:,0] / 255.).astype(np.uint8)
                cur_prompt_image_list.append(mask)
            else:
                gr.Warning(f"Number of {MASKS_PLACEHOLDER} is {input_text.count(MASKS_PLACEHOLDER)} \
                            not equal to 1, we do not record the scribble that you plot!")
    
    if point_mask is not None and isinstance(point_mask['point'],np.ndarray):
        assert not isinstance(input_image,dict)
        if input_text.count(MASKS_PLACEHOLDER) == 1:
            cur_prompt_image_list.append(point_mask['point'])
        else:
            gr.Warning(f"Number of {MASKS_PLACEHOLDER} is {input_text.count(MASKS_PLACEHOLDER)} \
                           not equal to 1, we do not record the point that you plot!")

    if input_image is not None:
        if isinstance(input_image,np.ndarray):
            state['input_data'].add_image(input_image)
        elif isinstance(input_image,dict):
            state['input_data'].add_image(input_image['image'])
    state['input_data'].append_message_question(input_text,system_value,cur_prompt_image_list)
    state['input_data'].add_one_conversation()
    state['input_data_okapi'].add_dataset(state['input_data'])
    state['dataloader'] = DataLoader(batch_size=1,num_workers=0,dataset=state['input_data_okapi'],collate_fn=okapi_collate_fn)


    return "", chatbot, state, prompt_image_list,radio,input_image


def submit_step2(chatbot, state,prompt_image_list,radio,temperature,top_p,top_k):


    if radio == "":
        raise gr.Error("You must set a valid task first!")
    
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

    output_seq = decode_generate_ids(tokenizer,output['generate_ids'][0],skip_special_tokens=False)
    output_seq = output_seq.replace("<","\<").replace(">","\>")
    bot_message = output_seq
    print(output_seq)
    chatbot[-1][1] = ""

    if output['decoder_outputs'] is not None:
        if 'box' in output['decoder_outputs'].keys():
            boxes_output = output['decoder_outputs']['box']['preds'][0].cpu().tolist()
            state['boxes_output'] = boxes_output
        if 'mask' in output['decoder_outputs'].keys():
            masks_output = output['decoder_outputs']['mask']['preds'][0].float().cpu().numpy().tolist()
            state['masks_output'] = masks_output
        try:
            labels = get_cot_elements(output_seq.replace("\\",""),['<REF>'])
            new_labels = []
            for label, num in zip(labels[0],labels[2]):
                new_labels.extend([label.strip()]*int(num))
        except:
            new_labels = None
        state['labels'] = new_labels

    for character in bot_message:
        chatbot[-1][1] += character
        time.sleep(0.005)
        yield chatbot
    
    return chatbot

def submit_step3(state,input_image,output_image,threshold=0.4):
    output_image = None
    if input_image is not None:
        if isinstance(input_image,dict):
            input_image = input_image['image']
    input_image = Image.fromarray(input_image)    

    boxes_output = state['boxes_output']
    masks_output = state['masks_output']

    if boxes_output != []:
        boxes_denorm = []
        for box in boxes_output:
            box_denorm = denorm_box_xywh_square2origin(box,input_image.width,input_image.height)
            boxes_denorm.append(box_denorm)
        output_image = visualize_box(input_image,boxes_denorm,labels=state['labels'])
    elif masks_output != []:
        masks_resize = [mask_square2origin(torch.tensor(mask),input_image.width,input_image.height) for mask in masks_output]
        masks_resize = [(mask > threshold).cpu().numpy().astype(np.uint8) for mask in masks_resize]
        output_image = visualize_mask(input_image,masks_resize)
    if output_image is not None:
        output_image = Image.fromarray(output_image).resize((600,330))

    state['boxes_output'] = []
    state['masks_output'] = []

    return output_image


def clear_states(preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state):
    return None,None,{'point':None},[],[],None,{
                                        'previous_system':None,
                                        'input_data':SingleInferDataset(),
                                        'input_data_okapi':BUILDER.build(cfg.infer_dataset),
                                        'dataloader': None,
                                        'prompt_input':None,
                                        'boxes_output':[],
                                        'masks_output':[],
                                        'labels':None}
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



title_markdown = ("""
<p align="center">
  <a href="#">
<img src="https://i2.mjj.rip/2024/06/12/c1f69496f89fee9eca4a5a1c279e373b.png" alt="Logo"></a>
</p>
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
    selected_points = gr.State(value=None)
    point_mask = gr.State(value={'point':None})
    prompt_image_list = gr.State(value=[])
    task_state = gr.State(value={'previous_system':None,
                                 'input_data':SingleInferDataset(),
                                 'input_data_okapi':BUILDER.build(cfg.infer_dataset),
                                 'dataloader': None,
                                 'boxes_output':[],
                                 'masks_output':[],
                                 'labels':None})
    example_list_image = [
        [os.path.join(os.path.dirname(__file__), "assets/dog.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/fishing.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/rec_bear.png")],
        [os.path.join(os.path.dirname(__file__), "assets/woman.jpeg")],
        [os.path.join(os.path.dirname(__file__), "assets/view.jpg")],
    ]


    example_list_1 = [
         ["VQA","Describe the main features of the image."],
         ["VQA","Please describe the image in more details."],
         ["VQA","Where is the dog?"],
         ["Detection","Detect objects in this image."],
         ["Segmentation","segment objects in this image."],
         ["Grounding Detection", "Please identify the position of young boy in the image and give the bounding box coordinates."],
         ["Grounding Segmentation", "Can you segment bears in the image and provide the masks for this class?"],
         ["Gcg Detection","Can you provide a description of the image and include the coordinates [x0,y0,x1,y1] for each mentioned object?"],
         ["Gcg Segmentation","Please explain what's happening in the photo and give masks for the items you reference."],
    ]

    example_list_2 = [
         ["VQA","Describe in detail how the area <masks> is represented in the image"],
         ["Grounding Detection", "Please detect the object using the guidance of <masks> in the image and describe the bounding box you create."],
         ["Grounding Segmentation", "Can you produce a detailed mask for the area indicated by the region <masks> in the image?"],
    ]

    example_list_3 = [
         ["VQA","Describe in detail how the area <masks> is represented in the image"],
         ["Grounding Detection", "Please detect the object using the guidance of <masks> in the image and describe the bounding box you create."],
         ["Grounding Segmentation", "Can you produce a detailed mask for the area indicated by the region <masks> in the image?"],
    ]

    example_list_4 = [
         ["VQA","Describe in detail how the area <masks> is represented in the image"],
         ["Grounding Detection", "Please detect the object using the guidance of <masks> in the image and describe the bounding box you create."],
         ["Grounding Segmentation", "Can you produce a detailed mask for the area indicated by the region <masks> in the image?"],
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
            with gr.TabItem("Input Image"):
                with gr.Row():
                    with gr.Column():
                        input_img_1 = ImageSketcher(type="numpy", label="Input Image", height=550)
                        
                        radio_1 = gr.Dropdown(
                            label="Task", choices=["VQA", "Detection", "Segmentation", 
                                                "Grounding Detection", "Grounding Segmentation", 
                                                "Gcg Detection", "Gcg Segmentation"],
                            value='VQA',info='Select valid task for Ladon!')                
                        input_text_1 = gr.Textbox(label="Input Instruction",placeholder='Message Ladon')
                        submit_button_1 = gr.Button("Submit", variant="primary")
                        example_data_1 = gr.Dataset(
                            label="Image Examples",
                            components=[input_img_1],
                            samples=example_list_image,
                        )
                        gr.Examples(
                            examples=example_list_1,
                            inputs=[radio_1,input_text_1],
                        )

            with gr.TabItem("Point"):
                with gr.Row():
                    with gr.Column():
                        input_img_2 = ImageSketcher(type="numpy", label="Input Image", height=550)
                        radio_2 = gr.Dropdown(label="Task", choices=["VQA","Grounding Detection", 
                                                                    "Grounding Segmentation"],
                                                value='VQA',info='Select valid task for Ladon!')
                        input_text_2 = gr.Textbox(label="Input Instruction",placeholder='Message Ladon') 
                        submit_button_2 = gr.Button("Submit", variant="primary")
                        example_data_2 = gr.Dataset(
                            label="Image Examples",
                            components=[input_img_2],
                            samples=example_list_image,
                        )
                        gr.Examples(
                            examples=example_list_2,
                            inputs=[radio_2,input_text_2],
                        )

            with gr.TabItem("Box"):
                with gr.Row():
                    with gr.Column():
                        input_img_3 = ImageSketcher(type="numpy", tool='boxes', label='Input Image', height=550)
                        radio_3 = gr.Dropdown(label="Task", choices=["VQA","Grounding Detection", 
                                                                    "Grounding Segmentation"],
                                                value='VQA',info='Select valid task for Ladon!')
                        input_text_3 = gr.Textbox(label="Input Instruction",placeholder='Message Ladon') 
                        submit_button_3 = gr.Button("Submit", variant="primary")
                        example_data_3 = gr.Dataset(
                            label="Image Examples",
                            components=[input_img_3],
                            samples=example_list_image,
                        )
                        gr.Examples(
                            examples=example_list_3,
                            inputs=[radio_3,input_text_3],
                        )


            with gr.TabItem("Scribble"):
                with gr.Row():
                    with gr.Column():
                        input_img_4 = ImageSketcher(type="numpy",tool="sketch", brush_radius=5,
                                                source="upload", label="Input Image", height=550)
                        radio_4 = gr.Dropdown(label="Task", choices=["VQA","Grounding Detection", 
                                                                    "Grounding Segmentation"],
                                            value='VQA',info='Select valid task for Ladon!')
                        input_text_4 = gr.Textbox(label="Input Instruction",placeholder='Message Ladon')
                        submit_button_4 = gr.Button("Submit", variant="primary")                      
                        example_data_4 = gr.Dataset(
                            label="Image Examples",
                            components=[input_img_4],
                            samples=example_list_image,
                        )
                        gr.Examples(
                            examples=example_list_4,
                            inputs=[radio_4,input_text_4],
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
                    chatbot = gr.Chatbot(height=600)
                    output_image = gr.Image(label="Output image", height=336, interactive=False)             
                    clear_button = gr.Button("🗑 Clear Button")
                    gr.Markdown(descrip_chatbot)

    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)


    
    clear_button.click(clear_states,
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state],
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state]).then(
        lambda: None,
        None,
        None,
        _js="() => {document.body.innerHTML='<h1 style=\"font-family:monospace;margin-top:20%;color:lightgray;text-align:center;\">Reloading...</h1>'; setTimeout(function(){location.reload()},2000); return []}",
    )
    
    
    input_img_1.upload(
        init_image,
        [input_img_1],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    input_img_2.upload(
        init_image,
        [input_img_2],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    input_img_3.upload(
        init_image,
        [input_img_3],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    input_img_4.upload(
        init_image,
        [input_img_4],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    input_img_2.select(
        img_select_point,
        [preprocessed_img, selected_points, point_mask],
        [input_img_2, point_mask],
    )


    submit_button_1.click(submit_step1,
                        [input_text_1,input_img_1,chatbot,radio_1,task_state,prompt_image_list],
                        [input_text_1,chatbot,task_state,prompt_image_list]).then(
                        submit_step2,
                        [chatbot,task_state,prompt_image_list,radio_1,temp_slider,top_p_slider,top_k_slider],
                        [chatbot]
                        ).then(
                        submit_step3,
                        [task_state,input_img_1,output_image],
                        [output_image]
                        )
    submit_button_2.click(submit_step1,
                        [input_text_2,input_img_2, chatbot,radio_2,task_state,prompt_image_list,point_mask],
                        [input_text_2,chatbot,task_state,prompt_image_list]).then(
                        submit_step2,
                        [chatbot,task_state,prompt_image_list,radio_2,temp_slider,top_p_slider,top_k_slider],
                        [chatbot]
                        ).then(
                        submit_step3,
                        [task_state,input_img_2,output_image],
                        [output_image]
                        )
    submit_button_3.click(submit_step1,
                        [input_text_3,input_img_3,chatbot,radio_3,task_state,prompt_image_list],
                        [input_text_3,chatbot,task_state,prompt_image_list]).then(
                        submit_step2,
                        [chatbot,task_state,prompt_image_list,radio_3,temp_slider,top_p_slider,top_k_slider],
                        [chatbot]
                        ).then(
                        submit_step3,
                        [task_state,input_img_3,output_image],
                        [output_image]
                        )
    submit_button_4.click(submit_step1,
                        [input_text_4,input_img_4,chatbot,radio_4,task_state,prompt_image_list],
                        [input_text_4,chatbot,task_state,prompt_image_list]).then(
                        submit_step2,
                        [chatbot,task_state,prompt_image_list,radio_4,temp_slider,top_p_slider,top_k_slider,],
                        [chatbot]
                        ).then(
                        submit_step3,
                        [task_state,input_img_4,output_image],
                        [output_image]
                        )


    example_data_1.click(clear_states,
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state],
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state]).then(
        init_image,
        [example_data_1],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    example_data_2.click(clear_states,
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state],
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state]).then(
        init_image,
        [example_data_2],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    example_data_3.click(clear_states,
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state],
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state]).then(
        init_image,
        [example_data_3],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

    example_data_4.click(clear_states,
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state],
                         [preprocessed_img,selected_points,point_mask,prompt_image_list,chatbot,output_image,task_state]).then(
        init_image,
        [example_data_4],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            selected_points,
        ],
    )

demo.queue().launch(
    debug=True,server_port=6229,
)
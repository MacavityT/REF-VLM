from typing import List
import gradio as gr
import numpy as np
import cv2
import os
import time
import random
from xtuner.utils import PROMPT_TEMPLATE

SYS_TEMPLATE_OKAPI = {
    'VQA': 'Task Command:\n- task name: vqa\n- answer element: sentence\n- unit: None',
    'Detection': 'Task Command:\n- task name: detection\n- answer element: phrase, unit\n- unit: <Unit>box</Unit>\n',
    'Segmentation': 'Task Command:\n- task name: segmentation\n- answer element: phrase, unit\n- unit: <Unit>mask</Unit>\n',
    'Grounding Detection': 'Task Command:\n- task name: grounding_detection\n- answer element: phrase, unit\n- unit: <Unit>box</Unit>\n',
    'Grounding Segmentation': 'Task Command:\n- task name: grounding_segmentation\n- answer element: phrase, unit\n- unit: <Unit>mask</Unit>\n',
    'Gcg Detection': 'Task Command:\n- task name: gcg_detection\n- answer element: phrase, sentence, unit\n- unit: <Unit>box</Unit>\n',
    'Gcg Segmentation': 'Task Command:\n- task name: gcg_segmentation\n- answer element: phrase, sentence, unit\n- unit: <Unit>mask</Unit>\n',
}

class ImageSketcher(gr.Image):
    """
    Code is from https://github.com/jshilong/GPT4RoI/blob/7c157b5f33914f21cfbc804fb301d3ce06324193/gpt4roi/app.py#L365

    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'.
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(tool='boxes', **kwargs)

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


def img_select_point(original_img: np.ndarray, sel_pix: list, evt: gr.SelectData, prompt_image_list: list):
    img = original_img.copy()
    sel_pix.clear()
    sel_pix.append((evt.index, 1))  # append the foreground_point
    point_mask = np.zeros((img.shape[0],img.shape[1]))
    # draw points
    for point, label in sel_pix:
        cv2.circle(img, point, 3, (240, 240, 240), -1, 0)
        cv2.circle(img, point, 3, (30, 144, 255), 2, 0)
        cv2.circle(point_mask, point, 3, (255, 255, 255), -1, 0)
    sel_pix.clear()
    prompt_image_list.append((point_mask/255.).astype(np.uint8))
    return img, prompt_image_list


def img_select_box(input_image: dict, prompt_image_list: list):
    if 'boxes' in input_image.keys():
        box_mask = np.zeros((input_image['image'].shape[0],input_image['image'].shape[1]))
        boxes = input_image['boxes'][-1]
        pt1 = (int(boxes[0]),int(boxes[1]))
        pt2 = (int(boxes[2]),int(boxes[3]))
        cv2.rectangle(box_mask, pt1, pt2, (255, 255, 255), -1)
        prompt_image_list.append((box_mask/255.).astype(np.uint8))
    return prompt_image_list


def img_select_scribble(original_img: np.ndarray, sel_pix: list, evt: gr.SelectData):
    img = original_img.copy()
    # Clear the previous selection
    # sel_pix.clear()
    print(evt.index)
    # Append the new selected point
    sel_pix.append(evt.index)

    # Ensure we have exactly two points to draw the rectangle
    if len(sel_pix) > 2:
        pt1 = sel_pix[0]
        pt2 = sel_pix[1]

        # Draw a rectangle from pt1 to pt2
        cv2.rectangle(img, pt1, pt2, (0, 0, 0), 1)

        # After drawing the rectangle, clear the points for the next selection
        sel_pix.clear()

    return img




# def print_like_dislike(x: gr.LikeData):
#     print(x.index, x.value, x.liked)

     
def submit_step1(input_text, input_image, chatbot, radio, state, prompt_image_list):
    
    system_template = PROMPT_TEMPLATE['okapi']
    system_task_text = SYS_TEMPLATE_OKAPI[radio]
    if chatbot == []:
        system_text = system_template['SYSTEM_PREFIX'] + system_template['SYSTEM'].format(system=system_task_text)
    else:
        system_text = system_template['SYSTEM'].format(system=system_task_text)
    chatbot = chatbot + [[input_text, None]]
    state['prompt_input'] = system_text + system_template['INSTRUCTION'].format(input=input_text)

    if isinstance(input_image,dict):
        if 'previous_box_length' in state.keys():
            if state['previous_box_length'] < len(input_image['boxes']):
                prompt_image_list = img_select_box(input_image,prompt_image_list)
                state['previous_box_length'] += 1
        elif 'previous_mask_length' in state.keys():
            if state['previous_mask_length'] < len(input_image['mask']):
                print(input_image['mask'])
                prompt_image_list.append(input_image['mask'])
                state['previous_mask_length'] += 1

    return "", chatbot, state, prompt_image_list

def submit_step2(chatbot,state,prompt_image_list):

    print(prompt_image_list)
    bot_message = state['prompt_input']
    chatbot[-1][1] = ""
    
    for character in bot_message:
        chatbot[-1][1] += character
        time.sleep(0.05)
        yield chatbot
    
    return chatbot

# import debugpy
# debugpy.connect(('127.0.0.1', 5577))



with gr.Blocks(
    title="Ladon: Multi-Visual Tasks Multimodal Large Language Model"
) as demo:
    preprocessed_img = gr.State(value=None)
    selected_points = gr.State(value=None)
    prompt_image_list = gr.State(value=[])

    example_list2 = [
        [os.path.join(os.path.dirname(__file__), "assets/dog.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/fishing.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/rec_bear.png")],
        [os.path.join(os.path.dirname(__file__), "assets/woman.jpeg")],
    ]

    example_list = [
        ["Could you please give me a detailed description of the image?"],
        ["Could you please give me a detailed description of the image?"],
        ["Could you please give me a detailed description of the image?"],
        ["Could you please give me a detailed description of the image?"],
    ]

    example_list3 = [
        [os.path.join(os.path.dirname(__file__), "assets/dog.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/fishing.jpg")],
        [os.path.join(os.path.dirname(__file__), "assets/rec_bear.png")],
        [os.path.join(os.path.dirname(__file__), "assets/woman.jpeg")],
    ]

    example_list4 = [
        [
            os.path.join(os.path.dirname(__file__), "assets/dog.jpg"),
            "Could you provide me with a detailed analysis of this photo? ",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/fishing.jpg"),
            "Could you provide me with a detailed analysis of this photo?",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
            "Could you provide me with a detailed analysis of this photo?",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/woman.jpeg"),
            "Could you provide me with a detailed analysis of this photo?",
        ],
    ]

    example_list5 = [
        [
            os.path.join(os.path.dirname(__file__), "assets/dog.jpg"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/fishing.jpg"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/woman.jpeg"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
    ]

    example_list6 = [
        [
            os.path.join(os.path.dirname(__file__), "assets/dog.jpg"),
            "Could you provide me with a detailed analysis of this photo? ",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/fishing.jpg"),
            "Could you provide me with a detailed analysis of this photo?",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
            "Could you provide me with a detailed analysis of this photo?",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/woman.jpeg"),
            "Could you provide me with a detailed analysis of this photo?",
        ],
    ]

    example_list7 = [
        [
            os.path.join(os.path.dirname(__file__), "assets/dog.jpg"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/fishing.jpg"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
        [
            os.path.join(os.path.dirname(__file__), "assets/woman.jpeg"),
            "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        ],
    ]

    descrip_detection = """
                    ### 💡 Tips:
                    
                    🧸 Upload an image, and you can click on the image to select the area of interest.
                    
                    🖱️ Then click the **Submit** button to generate detection and description accordingly.
            
                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                    
                    🔔 If you want to choose another area, just click another point on the image.

                    📌 Click the button ❎ to clear the current image.
    """

    descrip_segmentation = """
                    ### 💡 Tips:

                    🧸 Upload an image, and you can pull a frame on the image to select the area of interest.

                    🖱️ Then click the **Submit** button to generate segmentation and description accordingly.

                    🔔 If you want to choose another area or switch to another photo, click the button ↪️ first.

                    ❗️ If there are more than one box, the last one will be chosen.

                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.

                    📌 Click the button **Clear Image** to clear the current image. 
    
    """

    descrip_vqa = """
                    ### 💡 Tips:

    
    """

    descrip_gcg_detection = """
                    ### 💡 Tips:
    
    """
    descrip_gcg_segmentation = """
                    ### 💡 Tips:
    
    """
    descrip_grounding_detection = """
                    ### 💡 Tips:

                    🧸 Upload an image, and you can pull a frame on the image to select the area of interest.

                    🖱️ Then click the **Generate box and description** button to generate segmentation and description accordingly.

                    🔔 If you want to choose another area or switch to another photo, click the button ↪️ first.

                    ❗️ If there are more than one box, the last one will be chosen.

                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.

                    📌 Click the button **Clear Image** to clear the current image. 
    
    """
    descrip_grounding_segmentation = """
                    ### 💡 Tips:

                    🧸 Upload an image, and you can pull a frame on the image to select the area of interest.

                    🖱️ Then click the **Generate mask and description** button to generate segmentation and description accordingly.

                    🔔 If you want to choose another area or switch to another photo, click the button ↪️ first.

                    ❗️ If there are more than one box, the last one will be chosen.

                    🔖 In the bottom left, you can choose description with different levels of detail. Default is short description. 
                    
                    ⌛️ It takes about 1~ seconds to generate the segmentation result and the short description. The detailed description my take a longer time to 2~ seconds. The concurrency_count of queue is 1, please wait for a moment when it is crowded.

                    📌 Click the button **Clear Image** to clear the current image. 
    
    """
    with gr.Row():
        gr.HTML(
            """
            <h1 style="text-align: center; font-weight: 800; font-size: 2rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
            Ladon: Multi-Visual Tasks Multimodal Large Language Model
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
                        input_img_1 = gr.Image(type="numpy", label="Input Image", height=550)
                        radio_1 = gr.Radio(
                            label="Type", choices=["VQA", "Detection", "Segmentation", 
                                                   "Grounding Detection", "Grounding Segmentation", 
                                                   "Gcg Detection", "Gcg Segmentation"])
                        state_1 = gr.State(value={'prompt_input':None})                   
                        input_text_1 = gr.Textbox(label="Input Instruction")
                        submit_button_1 = gr.Button("Submit", variant="primary")
                        example_data_1 = gr.Dataset(
                            label="Examples", components=[input_text_1], samples=example_list
                        )

            with gr.TabItem("Point"):
                with gr.Row():
                    with gr.Column():
                        input_img_2 = gr.Image(type="numpy", label="Input Image", height=550)
                        radio_2 = gr.Radio(label="Type", choices=["VQA","Grounding Detection", 
                                                                    "Grounding Segmentation"])
                        state_2 = gr.State(value={'prompt_input':None})
                        input_text_2 = gr.Textbox(label="Input Instruction") 
                        submit_button_2 = gr.Button("Submit", variant="primary")
                        example_data_2 = gr.Dataset(
                            label="Examples",
                            components=[input_img_2, input_text_2],
                            samples=example_list6,
                        )


            with gr.TabItem("Box"):
                with gr.Row():
                    with gr.Column():
                        input_img_3 = ImageSketcher(type="numpy", label='Input Image', height=550)
                        radio_3 = gr.Radio(label="Type", choices=["VQA","Grounding Detection", 
                                                                    "Grounding Segmentation"])
                        state_3 = gr.State(value={'prompt_input':None,'previous_box_length':0})
                        input_text_3 = gr.Textbox(label="Input Instruction") 
                        submit_button_3 = gr.Button("Submit", variant="primary")
                        example_data_3 = gr.Dataset(
                            label="Examples",
                            components=[input_img_3, input_text_3],
                            samples=example_list7,
                        )


            with gr.TabItem("Scribble"):
                with gr.Row():
                    with gr.Column():
                        input_img_4 = gr.Image(type="numpy",tool="sketch", brush_radius=2,
                                                source="upload", label="Input Image", height=550)
                        radio_4 = gr.Radio(label="Type", choices=["VQA","Grounding Detection", 
                                                                    "Grounding Segmentation"])
                        state_4 = gr.State(value={'prompt_input':None,'previous_mask_length':0})
                        input_text_4 = gr.Textbox(label="Input Instruction")
                        submit_button_4 = gr.Button("Submit", variant="primary")                      
                        example_data_4 = gr.Dataset(
                            label="Examples", components=[input_text_2], samples=example_list3
                        )



        with gr.Column():

            chatbot = gr.Chatbot(height=600)
            # chatbot.like(print_like_dislike, None, None)
            output_mask = gr.Image(label="Output image", height=300, interactive=False)             
            clear_button = gr.Button("🗑 Clear Button")
            gr.Markdown(descrip_grounding_detection)

    



    clear_button.click(lambda: None, [], [input_text_1,input_text_2,input_text_3,input_text_4,
                                        preprocessed_img,input_img_1,input_img_2,input_img_3,input_img_4,
                                        selected_points,prompt_image_list]).then(
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
        [preprocessed_img, selected_points, prompt_image_list],
        [input_img_2, prompt_image_list],
    )

    # input_img_2.select(
    #     img_select,
    #     [preprocessed_img, selected_points, radio_2, ground_det_state],
    #     [input_img_2, ground_det_state],
    # )

    # input_img_3.select(
    #     img_select,
    #     [preprocessed_img, selected_points, radio_3, ground_seg_state],
    #     [input_img_3, ground_seg_state],
    # )


    submit_button_1.click(submit_step1,
                          [input_text_1,input_img_1,chatbot,radio_1,state_1,prompt_image_list],
                          [input_text_1,chatbot,state_1,prompt_image_list]).then(
                          submit_step2,
                          [chatbot,state_1,prompt_image_list],
                          [chatbot]
                        )
    submit_button_2.click(submit_step1,
                          [input_text_2,input_img_2, chatbot,radio_2,state_2,prompt_image_list],
                          [input_text_2,chatbot,state_2,prompt_image_list]).then(
                          submit_step2,
                          [chatbot,state_2,prompt_image_list],
                          [chatbot]
                        )
    submit_button_3.click(submit_step1,
                          [input_text_3,input_img_3,chatbot,radio_3,state_3,prompt_image_list],
                          [input_text_3,chatbot,state_3,prompt_image_list]).then(
                          submit_step2,
                          [chatbot,state_3,prompt_image_list],
                          [chatbot]
                        )
    submit_button_4.click(submit_step1,
                          [input_text_4,input_img_4,chatbot,radio_4,state_4,prompt_image_list],
                          [input_text_4,chatbot,state_4,prompt_image_list]).then(
                          submit_step2,
                          [chatbot,state_4,prompt_image_list],
                          [chatbot]
                        )


    example_data_1.click(
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


demo.queue().launch(
    debug=True,
)
from typing import List
import gradio as gr
import numpy as np
import cv2
import os
import time
import random

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
        preprocessed_img,
        preprocessed_img,
        preprocessed_img,
        [],
        None,
    )


def vqa(Input_Image=None, prompt_image=None, Input_text=None):
    print(Input_Image)
    if len(Input_Image) == 0 and prompt_image == None and Input_text == None:
        raise "Please input at least one text or one image."
    else:
        return 'yes'


def grounding_detection(Input_Image, Input_text=None, prompt_image=None):
    if Input_Image == None:
        raise "Please input a image"
    elif Input_text == None and prompt_image == None:
        raise "Please input at least one text or one image prmopt."
    # This is a placeholder function
    else:
        system = ""
        Output = "/code/okapi-mllm/demo/assets/rec_bear.png"
        text = "yes"
        return Output, text


def grounding_segmentation(Input_Image, Input_text=None, prompt_image=None):
    if Input_Image == None:
        raise "Please input a image"
    elif Input_text == None and prompt_image == None:
        raise "Please input at least one text or one image prmopt."
    # This is a placeholder function
    else:
        system = ""
        Output = "/code/okapi-mllm/demo/assets/rec_bear.png"
        text = "yes"
        return Output, text


def gcg_detection(Input_Image, Input_Text):
    if Input_Image == None or Input_Text == None:
        raise "Please input a image and a text"
    else:
        return Input_Image, Input_Text


def gcg_segmentation(Input_Image, Input_Text):
    if Input_Image == None or Input_Text == None:
        raise "Please input a image and a text"
    else:
        return Input_Image, Input_Text


def detection(Input_Image, Input_Text):
    if Input_Image == None or Input_Text == None:
        raise "Please input a image and a text"
    else:
        return Input_Image, Input_Text


def segmentation(Input_Image, Input_Text):
    if Input_Image == None or Input_Text == None:
        raise "Please input a image and a text"
    else:
        return Input_Image, Input_Text


def img_select_point(original_img: np.ndarray, sel_pix: list, evt: gr.SelectData):
    img = original_img.copy()
    sel_pix.clear()
    sel_pix.append((evt.index, 1))  # append the foreground_point
    tmp = []
    # draw points
    for point, label in sel_pix:
        cv2.circle(img, point, 3, (240, 240, 240), -1, 0)
        cv2.circle(img, point, 3, (30, 144, 255), 2, 0)
        tmp.append(evt.index)
    sel_pix.clear()
    return img, tmp


def img_select_box(original_img: np.ndarray, sel_pix: list, evt: gr.SelectData):
    img = original_img.copy()

    # Append the new selected point
    sel_pix.append(evt.index)
    tmp = []
    # Ensure we have exactly two points to draw the rectangle
    if len(sel_pix) == 2:
        pt1 = sel_pix[0]
        pt2 = sel_pix[1]

        # Draw a rectangle from pt1 to pt2
        cv2.rectangle(img, pt1, pt2, (0, 0, 0), 1)
        tmp.append(pt1, pt2)
        # After drawing the rectangle, clear the points for the next selection
        sel_pix.clear()

    return img, tmp


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


def img_select(original_img: np.ndarray, sel_pix: list, evt: gr.SelectData, task):
    if task == "point":
        img, prompt_img = img_select_point(original_img, sel_pix, evt)
        return img, prompt_img
    elif task == "box":
        img, prompt_img = img_select_box(original_img, sel_pix, evt)
        return img, prompt_img


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    bot_message = random.choice(["你好吗？", "我爱你", "我很饿"])
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.05)
        yield history
     
def submit_step1(input_text,chatbot, task):
    chatbot = chatbot + [[input_text, None]]
    print(f"Task is: {task}")
    return "", chatbot

def submit_step2(input_text,chatbot):

    bot_message = random.choice(["你好吗？", "我爱你", "我很饿"])
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
    prompt_img = gr.State(value=None)

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
            with gr.TabItem("VQA"):
                with gr.Row():
                    with gr.Column():
                        input_img_1 = gr.Image(type="numpy", label="Input Image", height=550)
                        radio_1 = gr.Radio(
                            label="Type", choices=["point", "box", "scribble"], value="point"
                        )                        
                        input_text_1 = gr.Textbox(label="Input Instruction")
                        vqa_state = gr.State(value="vqa")
                        submit_button_1 = gr.Button("Submit", variant="primary")
                        example_data_1 = gr.Dataset(
                            label="Examples", components=[input_text_1], samples=example_list
                        )

            with gr.TabItem("Grounding_Detection"):
                with gr.Row():
                    with gr.Column():
                        input_img_2 = gr.Image(type="numpy", label="Input Image", height=550)
                        radio_2 = gr.Radio(
                            label="Type", choices=["point", "box", "scribble"], value="point"
                        )
                        input_text = gr.Textbox(label="Input Instruction")
                        submit_button_2 = gr.Button("Submit", variant="primary")                      
                        example_data_2 = gr.Dataset(
                            label="Examples", components=[input_text], samples=example_list3
                        )

            with gr.TabItem("Grounding_Segmentation"):
                with gr.Row():
                    with gr.Column():
                        input_img_3 = gr.Image(type="numpy", label="Input Image", height=550)
                        radio_3 = gr.Radio(
                            label="Type", choices=["point", "box", "scribble"], value="point"
                        )
                        input_text = gr.Textbox(label="Input Instruction")
                        submit_button_3 = gr.Button("Submit", variant="primary")
                        example_data_3 = gr.Dataset(
                            label="Examples", components=[input_img_2], samples=example_list2
                        )

            with gr.TabItem("GCG_Detection"):
                with gr.Row():
                    with gr.Column():
                        input_img_4 = gr.Image(type="numpy", label="Input Image", height=550)
                        input_text = gr.Textbox(label="Input Instruction")
                        submit_button_4 = gr.Button("Submit", variant="primary")
                        example_data_4 = gr.Dataset(
                            label="Examples",
                            components=[input_img_4, input_text],
                            samples=example_list4,
                        )

            with gr.TabItem("GCG_Segmentation"):
                with gr.Row():
                    with gr.Column():
                        input_img_5 = gr.Image(type="numpy", label="Input Image", height=550)
                        input_text = gr.Textbox(label="Input Instruction")
                        submit_button_5 = gr.Button("Submit", variant="primary")
                        example_data_5 = gr.Dataset(
                            label="Examples",
                            components=[input_img_5, input_text],
                            samples=example_list5,
                        )

            with gr.TabItem("Detection"):
                with gr.Row():
                    with gr.Column():
                        input_img_6 = gr.Image(type="numpy", label="Input Image", height=550)
                        input_text = gr.Textbox(label="Input Instruction")
                        submit_button_6 = gr.Button("Submit", variant="primary")
                        example_data_6 = gr.Dataset(
                            label="Examples",
                            components=[input_img_6, input_text],
                            samples=example_list6,
                        )


            with gr.TabItem("Segmentation"):
                with gr.Row():
                    with gr.Column():
                        input_img_7 = gr.Image(type="numpy", label="Input Image", height=550)
                        input_text = gr.Textbox(label="Input Instruction")
                        submit_button_7 = gr.Button("Submit", variant="primary")
                        example_data_7 = gr.Dataset(
                            label="Examples",
                            components=[input_img_7, input_text],
                            samples=example_list7,
                        )


        with gr.Column():

            chatbot = gr.Chatbot(show_copy_button=True,height=600)
            # input_text.submit(user, [input_text, chatbot], [input_text, chatbot]).then(
            #     bot, chatbot, chatbot
            # )
            chatbot.like(print_like_dislike, None, None)
            output_mask = gr.Image(label="Output image", height=300, interactive=False)             
            clear_button = gr.Button("🗑 Clear Button")
            gr.Markdown(descrip_grounding_detection)

    
    submit_button_1.click(submit_step1,[input_text_1,chatbot,vqa_state],[input_text_1,chatbot]).then(
        submit_step2,[input_text_1,chatbot],[chatbot]
    )


    clear_button.click(lambda: None, [], [input_img_1, input_text]).then(
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
            input_img_5,
            input_img_6,
            input_img_7,
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
            input_img_5,
            input_img_6,
            input_img_7,
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
            input_img_5,
            input_img_6,
            input_img_7,
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
            input_img_5,
            input_img_6,
            input_img_7,
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
            input_img_5,
            input_img_6,
            input_img_7,
            selected_points,
        ],
    )

    input_img_5.upload(
        init_image,
        [input_img_5],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            input_img_5,
            input_img_6,
            input_img_7,
            selected_points,
        ],
    )

    input_img_6.upload(
        init_image,
        [input_img_6],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            input_img_5,
            input_img_6,
            input_img_7,
            selected_points,
        ],
    )

    input_img_7.upload(
        init_image,
        [input_img_7],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            input_img_5,
            input_img_6,
            input_img_7,
            selected_points,
        ],
    )

    input_img_1.select(
        img_select,
        [preprocessed_img, selected_points, radio_1],
        [input_img_1, prompt_img],
    )

    input_img_3.select(
        img_select,
        [preprocessed_img, selected_points, radio_3],
        [input_img_3, prompt_img],
    )

    input_img_2.select(
        img_select,
        [preprocessed_img, selected_points, radio_2],
        [input_img_2, prompt_img],
    )

    # submit_button_1.click(vqa, [input_img_1, prompt_img, input_text_1], output_text_1)
    # submit_button_4.click(
    #     gcg_detection, [input_img_4, input_text_4], [output_mask_4, output_text_4]
    # )
    # submit_button_5.click(
    #     gcg_segmentation, [input_img_5, input_text_5], [output_mask_5, output_text_5]
    # )
    # submit_button_6.click(
    #     detection, [input_img_6, input_text_6], [output_mask_6, output_text_6]
    # )
    # submit_button_7.click(
    #     segmentation, [input_img_7, input_text_7], [output_mask_7, output_text_7]
    # )

    example_data_1.click(
        init_image,
        [example_data_1],
        [
            preprocessed_img,
            input_img_1,
            input_img_2,
            input_img_3,
            input_img_4,
            input_img_5,
            input_img_6,
            input_img_7,
            selected_points,
        ],
    )


demo.queue().launch(
    debug=True,
)
import os
from typing import List
import cv2
import gradio as gr
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
import time
import argparse
from mmengine.config import Config, DictAction
from demo.inferece import OkapiInference
from demo.utils import ImageBoxState, bbox_draw


default_chatbox = [("", "Please begin the chat.")]

def parse_args():
    parser = argparse.ArgumentParser(description='okapi demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file name or path.')
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

def shortcut_func(task_name, text):
    task_name = task_name[0]
    if task_name == "Grounding":
        return "Where is XXX in the image?"
    elif task_name == "Caption":
        return "Can you provide a description of the image and include the locations for each mentioned object?"
    elif task_name == "Explain":
        return text.strip()+" Please include object locations and explain."
    elif task_name == "Region Cap":
        return "What is region [0]?"
    return ""

def new_state():
    return {"ibs": ImageBoxState()}


def clear_fn(value):
    return "", default_chatbox, None, None, new_state()


def clear_fn2(value):
    return default_chatbox, None, new_state()




def main():
    args = parse_args()

    # load inference pipeline
    InferPipeline = OkapiInference(args.config)

    n_turn = 0
    history = None

    with gr.Blocks() as demo:
        gr.HTML(
            f"""
            <h1 align="center"><font color="#966661">NExT-Chat</font></h1>
            <p align="center">
                <a href='' target='_blank'>[Project]</a>
                <a href='' target='_blank'>[Paper]</a>
            </p>
            <h2>User Manual</h2>
            <ul>
            <li><p><strong>Grounding:</strong> Where is XXX in the &lt;image&gt;? </p></li>
            <li><p><strong>Caption with objects: </strong>Can you provide a description of the image &lt;image&gt; and include the locations for each mentioned object? </p></li>
            <li><p><strong>The model is default not to include obj locations at most time.</strong> </p></li>
            <li><p><strong>To let the model include object locations. You can add prompts like:</strong> </p></li>
                <ul>
                <li><p>Please include object locations and explain. </p></li>
                <li><p>Make sure to include object locations and explain. </p></li>
                <li><p>Please include object locations as much as possible. </p></li>
                </ul>
            <li><p><strong>Region Understanding:</strong> draw boxes and ask like "what is region [0]?" </p></li>

            <ul>
            """
        )

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Group():
                    input_shortcuts = gr.Dataset(components=[gr.Textbox(visible=False)], samples=[
                        ["Grounding"],
                        ["Caption"], ["Explain"], ["Region Cap"]], label="Shortcut Dataset")

                    input_text = gr.Textbox(label='Input Text',
                                            placeholder='Please enter text prompt below and press ENTER.')

                    with gr.Row():
                        input_image = gr.Image(type='filepath',label='Input Image')
                        out_imagebox = gr.Image(label="Parsed Sketch Pad")
                    input_image_state = gr.State(new_state())

                with gr.Row():
                    temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                    top_p = gr.Slider(maximum=1, value=0.7, minimum=0, label='Top P')
                    top_k = gr.Slider(maximum=100, value=5, minimum=1, step=1, label='Top K')

                with gr.Row():
                    run_button = gr.Button('Generate')
                    clear_button = gr.Button('Clear')

            with gr.Column(scale=4):
                output_text = gr.components.Chatbot(label='Multi-round conversation History',
                                                    value=default_chatbox).style(height=550)
                output_image = gr.Textbox(visible=False)

        input_shortcuts.click(fn=shortcut_func, inputs=[input_shortcuts, input_text], outputs=[input_text])

        run_button.click(fn=InferPipeline, inputs=[input_text,history,input_image,n_turn],
                         outputs=[input_text,output_text, history, n_turn])
        input_text.submit(fn=InferPipeline, inputs=[input_text,history,input_image,n_turn],
                          outputs=[input_text, output_text, history, n_turn])
        clear_button.click(fn=clear_fn, inputs=clear_button,
                           outputs=[input_text, output_text, input_image, out_imagebox, input_image_state])
        input_image.upload(fn=clear_fn2, inputs=clear_button, outputs=[output_text, out_imagebox, input_image_state])
        input_image.clear(fn=clear_fn2, inputs=clear_button, outputs=[output_text, out_imagebox, input_image_state])
        input_image.edit(
            fn=bbox_draw,
            inputs=[input_image, input_image_state],
            outputs=[out_imagebox, input_image_state],
            queue=False,
        )

        with gr.Row():
            gr.Examples(
                examples=[
                    [
                        os.path.join(os.path.dirname(__file__), "assets/dog.jpg"),
                        "Can you describe the image and include object locations?",
                        new_state(),
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/fishing.jpg"),
                        "A boy is sleeping on bed, is this correct? Please include object locations.",
                        new_state(),
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
                        "Where is the bear wearing the red decoration in the image?",
                        new_state(),
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/woman.jpeg"),
                        "What is the woman doing? Please include object locations.",
                        new_state(),
                    ],
                ],
                inputs=[input_image, input_text, input_image_state],
            )

    print("launching...")
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port, share=True)

if __name__ == "__main__":
    main()
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
import pickle
from PIL import Image
from transformers import AutoModelForCausalLM
from xtuner.utils import PROMPT_TEMPLATE,DEFAULT_IMAGE_TOKEN,VISUAL_PROMPT_PLACEHOLDER,BOV_TOKEN,EOV_TOKEN,VISUAL_REPRESENTATION_TOKEN
from xtuner.utils.constants import MASKS_PLACEHOLDER
from xtuner.dataset import OkapiDataset
from xtuner.dataset.collate_fns import okapi_collate_fn
from xtuner.dataset.map_fns.dataset_map_fns.okapi_map_fn_stage2 import get_cot_elements
from xtuner.dataset.utils import (visualize_box,
                                  visualize_mask,
                                  visualize_keypoints,
                                  visualize_all_keypoints,
                                  visualize_keypoints_pytorch,
                                  mask_square2origin,
                                  draw_label_type,
                                  denorm_box_xywh_square2origin,
                                  de_norm_keypoint_square2origin)
from inference import OkapiInference
from utils import SingleInferDataset
from mmengine.config import Config, DictAction
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import prepare_inputs_labels_for_multimodal



# import debugpy
# debugpy.connect(('127.0.0.1', 5577))


if __name__ == "__main__":
    config_path = "configs/sketch_okapi_7b_inference_stage2_decoder.py"
    save_name = "unfreeze"


        
    if not os.path.isfile(config_path):
        try:
            config = cfgs_name_path[config_path]
        except KeyError:
            raise FileNotFoundError(f'Config arg is not None but cannot find {config_path}')
    else:
        config = config_path
    cfg = Config.fromfile(config)

    # load model and dataset
    model = BUILDER.build(cfg.model)
    train_dataset = BUILDER.build(cfg.train_dataset)
    dataloader = DataLoader(train_dataset,collate_fn=okapi_collate_fn,batch_size=1,num_workers=0,shuffle=False)

    tokenizer = model.tokenizer
    model.cuda().eval()
    llm = AutoModelForCausalLM.from_pretrained(cfg.model_dir,attn_implementation="eager",trust_remote_code=True).to(model.llm.dtype).to(model.llm.device)
    llm.eval()
    print(cfg.model_dir)
    model.llm = llm


    # start inference
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            for key in data_batch['data'].keys():
                if isinstance(data_batch['data'][key],torch.Tensor):
                    data_batch['data'][key] = data_batch['data'][key].cuda()
            output = model(**data_batch,mode='tensor')
            break
    
    # get llm matrix
    llm_attn_matrix = torch.stack(output[0]['attentions']).cpu()
    llm_attn_matrix_mean = llm_attn_matrix.mean(dim=0).mean(dim=1)[0].float()
    # get vision token start
    input_ids = data_batch['data']['input_ids'][0]
    image_index = torch.where(input_ids<0)[0]
    print(image_index)
    vision_token_start = image_index
    append_id = input_ids.tolist()[:vision_token_start] + [-200 for _ in range(576)] + input_ids.tolist()[vision_token_start+1:]
    ref_indices = np.where(np.array(append_id) == 32009)[0]
    decode_labels = [tokenizer.decode(token, add_special_tokens=True).strip() for token in input_ids[:vision_token_start]] + \
        ["<image>" for _ in range(576)] + [tokenizer.decode(token, add_special_tokens=True).strip() for token in input_ids[vision_token_start+1:]]

    save_dict = {
        'attention_mean': llm_attn_matrix_mean,
        'attention': llm_attn_matrix,
        'input_ids': input_ids,
        'image_index': image_index,
        'ref_indices': ref_indices,
        'decode_labels': decode_labels
    }

    
    with open(f"{save_name}.pkl","wb") as f:
        pickle.dump(save_dict,f)
        f.close()


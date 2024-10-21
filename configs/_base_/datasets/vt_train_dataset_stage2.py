# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from dataset.map_fns import (
    vt_map_fn_stage2,
    vt_keypoint_map_fn,
    vt_template_map_fn_factory
)
from dataset import VTInstructDataset
from dataset.collate_fns import vt_collate_fn
from mmengine.dataset import DefaultSampler

with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336, clip_convnext_512

from utils import PROMPT_TEMPLATE
prompt_template = PROMPT_TEMPLATE.vd_cot

# Data configs
max_length = 2048 - 576 # use cutoff lens instead
cutoff_len = 2048
visual_hidden_size = 1024 # visual_encoder.config.hidden_size
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8

gc = dict(
    type='SubSet',
    portion=1/15,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['gc'],
)

grand_re_cap = dict(
    type='SubSet',
    portion=1/2,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_c'],
)


grand_cond_d = dict(
    type='SubSet',
    portion=1/13,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_cond_d'],
)

grand_cond_s = dict(
    type='SubSet',
    portion=3/4,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_cond_s'],
)



dataset_s1 = [

    # subsets
    gc,

    # general datasets
    train_all_dataset['flickr'],
    train_all_dataset['rec'],
    train_all_dataset['caption'],
    train_all_dataset['reg'],
    train_all_dataset['res_refcoco'],
    train_all_dataset['res_refcocoa'],
    train_all_dataset['res_refcocog'],

    # vqa v2
    train_all_dataset['vqav2_train'],
    train_all_dataset['vqae_train'],
    train_all_dataset['vqax_train'],

    # point qa
    train_all_dataset['point_local_b'],
    train_all_dataset['point_local_p'],
    train_all_dataset['point_twice_oq_bp'],
    train_all_dataset['point_twice_sq_bp'],
    train_all_dataset['point_twice_gq_bp'],
    train_all_dataset['point_v7w_p'],
    train_all_dataset['point_v7w_b'],
]



dataset_s2 = [


    # llava grounding
    train_all_dataset['llavag_reg'],
    train_all_dataset['llavag_gcg'],
    
    # png
    # train_all_dataset['png_gcg'],
    train_all_dataset['openpsg'],

    # instruct
    train_all_dataset['instruct'],

    # osprey
    train_all_dataset['osprey_partlevel'],
    train_all_dataset['osprey_shortform'],
    train_all_dataset['osprey_lvis'],
    train_all_dataset['osprey_conversations'],
    train_all_dataset['osprey_detailed'],

    # interact
    train_all_dataset['interact_reg'],
    train_all_dataset['interact_mask'],
    train_all_dataset['interact_box'],

    # grit
    train_all_dataset['grit_c_offline'],
    train_all_dataset['grit_d_offline'],
    train_all_dataset['grit_cond_d_offline'],
    train_all_dataset['grit_r_offline'],
    train_all_dataset['grit_g_offline'],
    train_all_dataset['grit_c_d_offline'],

    # grand
    grand_re_cap,
    grand_cond_d,
    grand_cond_s,
    train_all_dataset['grand_d'],
    train_all_dataset['grand_s'],
    train_all_dataset['grand_re_det'],
    train_all_dataset['grand_re_seg'],
    train_all_dataset['grand_c_d'],
    train_all_dataset['grand_c_s'],

]


dataset_args = dataset_s2 + dataset_s1

for dataset in dataset_args:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',2)
    else:
        dataset['stage'] = 2

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336

gc = dict(
    type='SubSet',
    portion=1/20,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['gc'],
)

caption = dict(
    type='SubSet',
    portion=1/3,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['caption'],
)


reg = dict(
    type='SubSet',
    portion=1/3,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['reg'],
)

grand_re_cap = dict(
    type='SubSet',
    portion=1/2,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_c'],
)

grand_det_seg = dict(
    type='SubSet',
    portion=1/10,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_d_s'],
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
    portion=1/2,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_cond_s'],
)

grand_re_det = dict(
    type='SubSet',
    portion=1/2,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_re_det'],
)

grand_re_seg = dict(
    type='SubSet',
    portion=1/2,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grand_re_seg'],
)

dataset_s1 = [

    # subsets
    gc,

    # general datasets
    train_all_dataset['flickr'],
    train_all_dataset['rec'],
    train_all_dataset['caption'],
    train_all_dataset['reg'],
    
    # vcr
    train_all_dataset['vcr_qc_rac'],
    train_all_dataset['vcr_qac_r'],

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
    train_all_dataset['png_gcg'],

    # instruct
    train_all_dataset['instruct'],

    # gpt gen
    train_all_dataset['gpt4gen_qbc'],
    train_all_dataset['gpt4gen_rd_qbc'],

    # osprey
    # train_all_dataset['osprey_partlevel'],
    # train_all_dataset['osprey_shortform'],
    # train_all_dataset['osprey_lvis'],
    # train_all_dataset['osprey_conversations'],
    # train_all_dataset['osprey_detailed'],

    # interact
    train_all_dataset['interact_reg'],
    train_all_dataset['interact_mask'],
    train_all_dataset['interact_box'],

    # grit
    train_all_dataset['grit_c'],
    train_all_dataset['grit_d'],
    train_all_dataset['grit_cond_d'],
    train_all_dataset['grit_r'],
    train_all_dataset['grit_g'],
    train_all_dataset['grit_c_d'],

    # grand
    grand_re_cap,
    grand_det_seg,
    grand_cond_d,
    grand_cond_s,
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

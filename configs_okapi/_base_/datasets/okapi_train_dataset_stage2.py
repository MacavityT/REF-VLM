# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336

gc = dict(
    type='SubSet',
    portion=1/15,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['gc'],
)

grit = dict(
    type='SubSet',
    portion=1/3,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['grit_combine_offline'],
)


dataset_args = [
    # subsets
    gc,
    grit,

    # general datasets
    train_all_dataset['flickr'],
    train_all_dataset['rec'],
    train_all_dataset['reg'],
    train_all_dataset['caption'],
    train_all_dataset['instruct'],

    # vqa v2
    train_all_dataset['vqav2_train'],
    train_all_dataset['vqae_train'],
    train_all_dataset['vqax_train'],

    # vcr
    train_all_dataset['vcr_q_ra'],
    train_all_dataset['vcr_qc_rac'],
    train_all_dataset['vcr_qac_r'],

    # point qa
    train_all_dataset['point_local_b'],
    train_all_dataset['point_local_p'],
    train_all_dataset['point_twice_oq_bp'],
    train_all_dataset['point_twice_sq_bp'],
    train_all_dataset['point_twice_gq_bp'],
    train_all_dataset['point_v7w_p'],
    train_all_dataset['point_v7w_b'],

    # gpt gen
    train_all_dataset['gpt4gen_qbc'],
    train_all_dataset['gpt4gen_rd_qbc'],

    # ospery
    train_all_dataset['ospery_partlevel'],
    train_all_dataset['ospery_shortform'],
    train_all_dataset['ospery_lvis'],
    train_all_dataset['ospery_conversations'],
    train_all_dataset['ospery_detailed'],

    # interact
    train_all_dataset['interact_mask'],
    train_all_dataset['interact_box'],

    # grand
    train_all_dataset['grand_mix'],
]


for dataset in dataset_args:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',2)
        dataset['cfg']['offline_processed_text_folder'] = ''
    else:
        dataset['offline_processed_text_folder'] = ''
        dataset['stage'] = 2

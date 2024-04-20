# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.dataset import (
    ConcatDataset,
    OkapiDataset,
    LLaVADataset
)
from xtuner.dataset.map_fns import (
    llava_map_fn, 
    okapi_map_fn, 
    template_map_fn_factory
)
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.utils import PROMPT_TEMPLATE

from mmengine.dataset import DefaultSampler
from mmengine.config import read_base
with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336

# Data
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 / 14)**2)

batch_size = 16  # per_device
dataloader_num_workers = 5

#region okapi dataset
gc = dict(
    type='SubSet',
    portion=1,
    do_shuffle=True,
    seed=42,
    cfg=train_all_dataset['gc'],
)

recvg = dict(
    type='SubSet',
    portion=1,
    do_shuffle=True,
    seed=43,
    cfg=train_all_dataset['recvg'],
)

dataset_args = [
    gc,
    recvg,
    # llava pretrain
    # train_all_dataset['llavacc3m'],
    # train_all_dataset['llavalcs'],
    # vqa
    train_all_dataset['vqav2_train'],
    train_all_dataset['vqae_train'],
    train_all_dataset['vqax_train'],
    # caption
    train_all_dataset['caption'],
    # ref
    train_all_dataset['rec'],
    train_all_dataset['reg'],
    # flickr
    train_all_dataset['flickr'],
    # vcr
    train_all_dataset['vcr_q_ra'],
    train_all_dataset['vcr_qc_rac'],
    train_all_dataset['vcr_qac_r'],
    # point_qa
    train_all_dataset['point_local_b'],
    train_all_dataset['point_local_p'],
    train_all_dataset['point_twice_oq_bp'],
    train_all_dataset['point_twice_sq_bp'],
    train_all_dataset['point_twice_gq_bp'],
    train_all_dataset['point_v7w_p'],
    train_all_dataset['point_v7w_b'],
]

okapi_dataset = dict(
    type=OkapiDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=vicuna_7b_path_tokenizer,
    dataset_map_fn=okapi_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)
#endregion

#region llava dataset
llava_dataset = dict(
    type=LLaVADataset,
    data_path=r"/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/blip_laion_cc_sbu_558k_filter.json",
    image_folder=r'/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/images',
    tokenizer=vicuna_7b_path_tokenizer,
    image_processor=clip_patch14_336['image_processor'],
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)
#endregion

train_dataset = dict(type=ConcatDataset, datasets=[llava_dataset, okapi_dataset])


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))



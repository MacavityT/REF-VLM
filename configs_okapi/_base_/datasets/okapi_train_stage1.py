# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from xtuner.dataset import OkapiDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE

from mmengine.config import read_base
with read_base():
    from ..models.all_tokenizers import vicuna_7b_path, vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336

# Data
data_root = './data/llava_data/'
data_path = data_root + 'LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
image_folder = data_root + 'LLaVA-Pretrain/images'
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 / 14)**2)

batch_size = 32  # per_device
dataloader_num_workers = 0

llava_dataset = dict(
    type=OkapiDataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=vicuna_7b_path_tokenizer,
    image_processor=clip_patch14_336['image_processor'],
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))




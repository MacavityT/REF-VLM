from functools import partial
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset.map_fns import (
    okapi_map_fn_stage2,
    okapi_template_map_fn_factory
)
import os
from xtuner.dataset.collate_fns import okapi_collate_fn
from transformers import AutoModel

from mmengine.config import read_base
with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336
    from ._base_.datasets.okapi_train_dataset_stage2 import *
    from ._base_.datasets.okapi_val_dataset_stage2 import *
    from ._base_.models.okapi_vicuna_7b import *

max_length = 2048 - 576 
cutoff_len = 2048
visual_hidden_size = 1024 # visual_encoder.config.hidden_size
vrt_length = 256
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
ref_length = 1
cot_weight = 1
vrt_weight = 1
model_dir = '/code/okapi-mllm/sketch_checkpoints/0821_iter31500'


projector = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'projector'),
    trust_remote_code=True,
)

vpt_encoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'vpt_encoder'),
    trust_remote_code=True,
)

visual_sync_tuner = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'visual_sync_tuner'),
    trust_remote_code=True,
)

box_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'box_decoder'),
    trust_remote_code=True,
)

mask_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'mask_decoder'),
    trust_remote_code=True,
)



infer_dataset = dict(
    type=OkapiDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=okapi_map_fn_stage2,
        args = dict(
            vrt_len=vrt_length, 
            ref_len=ref_length
        )
    ),
    template_map_fn=dict(
        type=okapi_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    mode='inference')



model = dict(
    type=OkapiModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'],
    projector=projector,
    vpt_encoder=vpt_encoder,
    visual_sync_tuner=visual_sync_tuner,
    visual_decoder=dict(
        box=box_decoder,
        mask=mask_decoder
    ))

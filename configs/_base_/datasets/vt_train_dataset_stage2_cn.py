# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset import LLaVADataset, ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from vt_plug.dataset.map_fns import vt_map_fn
from vt_plug.dataset import VTInstructDataset
from vt_plug.evaluation.metrics import ImgCapComputeMetrics, VQAComputeMetrics

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

recvg = dict(
    type='SubSet',
    portion=1/20,
    do_shuffle=True,
    seed=43,
    cfg=train_all_dataset['recvg'],
)

dataset_args_en = [
    gc,
    recvg,
    # llava pretrain
    train_all_dataset['llavacc3m'],
    train_all_dataset['llavalcs'],

    # llava instruct
    # train_all_dataset['instruct'],

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


dataset_args_cn = [
    # raw data
    train_all_dataset['product1m'],
    train_all_dataset['wukong'],

    # caption + chinese sft
    train_all_dataset['llavacc3m_cn'],
    train_all_dataset['instruct_cn'],
    train_all_dataset['flickr_caption_cn'],
    train_all_dataset['coco_caption_cn'],

    # emoji recognition
    train_all_dataset['cn_bqb_yi'],
    train_all_dataset['cn_bqb_claude'],
    train_all_dataset['cn_bqb_gemini'],
    train_all_dataset['cn_emo_claude'],
    train_all_dataset['cn_emo_gemini'],

    # ocr
    train_all_dataset['text_recognition_train'],
    train_all_dataset['text_train'],

    # vqa
    train_all_dataset['fm_iqa'],

    # cvlue
    train_all_dataset['cvlue_dialog'],
    train_all_dataset['cvlue_vqa'],
    train_all_dataset['cvlue_caption'],
    train_all_dataset['cvlue_rec'],
]

dataset_args = dataset_args_en + dataset_args_cn


for dataset in dataset_args:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',1)
    else:
        dataset['stage'] = 1

from mmengine.dataset import DefaultSampler
from xtuner.dataset import OkapiDataset,ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, okapi_map_fn
from xtuner.utils import PROMPT_TEMPLATE
from mmengine.config import read_base


with read_base():
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336
    from .test_rec_variant import test_rec_variant
    from .test_reg_variant import test_reg_variant
    from .train_grand_variant import train_grand_variant
    from .test_interact_variant import test_interact_variant
    from .test_flickr_variant import test_flickr_variant
    from .test_vqav2_variant import test_vqav2_variant
    from .test_pope_variant import test_pope_variant
    from .test_point_variant import test_point_variant
    from .test_res_variant import test_res_variant
    from .test_cocodet_variant import test_cocodet_variant
    from .test_cocogcg_variant import test_cocogcg_variant

test_cfg = dict(type='TestLoop')

test_all_dataset = dict(
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2017_val.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        template_name=r'image_cap',
    ),
    reg_box=dict(
        type='REGDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcocog_umd_test.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'REG',
        version='box',
        placeholders=('<image>','<objs>'),
        map_placeholders=dict(
            input=["<boxes>"],
        )        
    ),
    okvqa=dict(
        type='OKVQADataset',
        image_folder='/data/Aaronzhu/DatasetStage1/MSCOCO/2014/val',
        text_path='/data/Aaronzhu/DatasetStage1/OKVQA/okvqa_test.jsonl',
        has_annotation=False,
        template_name=r"VQA",
    ),
    **test_rec_variant,
    **train_grand_variant,
    **test_interact_variant,
    **test_reg_variant,
    **test_flickr_variant,
    **test_vqav2_variant,
    **test_pope_variant,
    **test_point_variant,
    **test_res_variant,
    **test_cocodet_variant,
    **test_cocogcg_variant,
)


for key,value in test_all_dataset.items():
    value.setdefault('stage',2)

from mmengine.config import read_base
from utils import PROMPT_TEMPLATE
from model import VTPlugModel

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

# Params
max_length = 2048 - 576 # use cutoff lens instead
cutoff_len = 2048
visual_hidden_size = 1024 # visual_encoder.config.hidden_size
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
prompt_template = PROMPT_TEMPLATE.vd_cot

# Datasets
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
    lvis_box=dict(
        type='LVISDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/LVIS/lvis_v1_val.json',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/LVIS/COCO2017/val2017',
        task_type='box',
        template_name=r'DET',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<boxes>"],
        ),
    ),
    lvis_box_test=dict(
        type='LVISTestDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/LVIS/lvis_v1_image_info_test_dev.json',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/LVIS/test2017',
        task_type='box',
        template_name=r'DET',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<boxes>"],
        ),
    ),
    test_ade20_with_instance=dict(
        type='ADE20k',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/ade20k_instance_val.json',  
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/images/validation',
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/annotations_instance/validation',
        target_type = 'instance',
        template_name=r'SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),
    ),
    test_cityscapes_instance=dict(
        text_path=r'/data/Aaronzhu/DatasetStage2and3/cityscapes/gtFine/val',  
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/cityscapes/leftImg8bit/val',
        placeholders=('<image>',),
        ratio=0.3,
        type='CityscapesInstance',
        split='val',
        target_type='instance',
        template_name=r'SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),
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

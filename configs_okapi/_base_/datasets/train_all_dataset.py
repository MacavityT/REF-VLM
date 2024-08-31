from mmengine.config import read_base

with read_base():
    from .train_gqa_variant import train_gqa_variant
    from .train_clevr_variant import train_clevr_variant
    from .train_point_variant import train_point_variant
    from .train_gptgen_variant import train_gptgen_variant
    from .train_vcr_variant import train_vcr_variant
    from .train_vqav2_variant import train_vqav2_variant
    from .train_vqaex_variant import train_vqaex_variant
    from .train_grit_variant import train_grit_variant
    from .train_grand_variant import train_grand_variant
    from .train_osprey_variant import train_osprey_variant
    from .train_interact_variant import train_interact_variant
    from .train_ade20k_variant import train_ade20k_variant
    from .train_pascal_variant import train_voc_variant
    from .train_cityscapes_variant import train_cityscapes_variant
    from .train_llavag_variant import train_llavag_variant
    from .train_png_variant import train_png_variant
    from .train_cocokeypoint_variant import train_cocokeypoints_variant
    from .train_reg_variant import train_reg_variant
    from .train_res_variant import train_res_variant

train_all_dataset = dict(
    flickr=dict(
        type='FlickrDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CWB_flickr30k_train.jsonl',  
        image_folder=r'/data/Aaronzhu/DatasetStage1/flickr30k/flickr30k-images',
        template_name=r'flickr30k',
        map_placeholders=dict(
            output=["<boxes>"],
        )
    ),
    rec=dict(
        type='RECDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_ref3_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'REC',
        map_placeholders=dict(
            output=["<boxes>"],
        )
    ),
    recvg=dict(
        type='RECDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/GC_genome196_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/Visual Genome',
        template_name=r'REC',
        map_placeholders=dict(
            output=["<boxes>"],
        )
    ),
    reg=dict(
        type='REGDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_ref3_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'REG',
        placeholders=('<image>','<objs>'),
        map_placeholders=dict(
            input=["<boxes>"],
        )
    ),
    gc=dict(
        type='GCDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/GC_genome196_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/Visual Genome',
        template_name=r'GC',
        placeholders=('<image>','<objs>'),
        map_placeholders=dict(
            input=["<boxes>"],
        )
    ),
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2014_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'image_cap',
    ),
    llavacc3m=dict(
        type='InstructDataset',
        text_path=r"/data/Aaronzhu/DatasetStage1/Shikra/llava_cc3m.jsonl",
        image_folder=r'/data/Aaronzhu/DatasetStage1/llava/llava/LLaVA-CC3M-Pretrain-595K/images',  
    ),
    llavalcs=dict(
        type='InstructDataset',
        text_path=r"/data/Aaronzhu/DatasetStage1/Shikra/blip_laion_cc_sbu_558k_filter.jsonl",
        image_folder=r'/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/images',
    ),
    instruct=dict(
        type='InstructMixDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/llava-instruct/llava_v1_5_mix665k_fliter_d.json',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/llava-instruct/images',
        offline_processed_text_folder=r'/data/Aaronzhu/DatasetStage2and3/llava-instruct/offline',
    ),
    coco_rem_mask=dict(
        type='COCOREMDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/COCO-ReM/instances_trainrem.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/COCO-ReM/train2017',
        task_type='mask',
        template_name=r'SEG',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<masks>"],
        ),
        offline_processed_text_folder='',
    ),
    coco_rem_box=dict(
        type='COCOREMDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/COCO-ReM/instances_trainrem.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/COCO-ReM/train2017',
        task_type='box',
        template_name=r'DET',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<boxes>"],
        ),
        offline_processed_text_folder='',
    ),
    kitti=dict(
        type='KITTIDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/KITTI/train',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/KITTI',
        template_name=r'Depth',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<depth>"],
        ),
        offline_processed_text_folder='',
    ),
    nyu=dict(
        type='NYUDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/NYU/nyu/depth',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/NYU/nyu/depth',
        template_name=r'Depth',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<depth>"],
        ),
        offline_processed_text_folder='',
    ),
    hrwsi=dict(
        type='HRWSIDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/HRWSI/HR-WSI/train/gts',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/HRWSI/HR-WSI/train/imgs',
        template_name=r'Depth',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<depth>"],
        ),
        offline_processed_text_folder='',
    ),
    **train_reg_variant,
    **train_gqa_variant,
    **train_clevr_variant,
    **train_point_variant,
    **train_gptgen_variant,
    **train_vcr_variant,
    **train_vqav2_variant,
    **train_vqaex_variant,
    **train_grit_variant,
    **train_grand_variant,
    **train_osprey_variant,
    **train_interact_variant,
    **train_ade20k_variant,
    **train_voc_variant,
    **train_cityscapes_variant,
    **train_llavag_variant,
    **train_png_variant,
    **train_cocokeypoints_variant,
    **train_res_variant,
)

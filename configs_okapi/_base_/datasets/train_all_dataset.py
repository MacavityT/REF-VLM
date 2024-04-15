from mmengine.config import read_base

with read_base():
    from .train_gqa_variant import train_gqa_variant
    from .train_clevr_variant import train_clevr_variant
    from .train_point_variant import train_point_variant
    from .train_gptgen_variant import train_gptgen_variant
    from .train_vcr_variant import train_vcr_variant
    from .train_vqav2_variant import train_vqav2_variant
    from .train_vqaex_variant import train_vqaex_variant


train_all_dataset = dict(
    flickr=dict(
        type='FlickrDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CWB_flickr30k_train.jsonl',  
        image_folder=r'/data/Aaronzhu/DatasetStage1/flickr30k/flickr30k-images',
        template_name=r'flickr30k',
    ),
    rec=dict(
        type='RECDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_ref3_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'REC',
    ),
    recvg=dict(
        type='RECDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/GC_genome196_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/Visual Genome',
        template_name=r'REC',
    ),
    reg=dict(
        type='REGDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_ref3_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'REG',
    ),
    gc=dict(
        type='GCDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/GC_genome196_train.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/Visual Genome',
        template_name=r'GC',
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
        image_folder=r'/data/Aaronzhu/DatasetStage1/llava/llava/LLaVA-CC3M-Pretrain-595K/images',  # TODO: zz make folder name mistake
    ),
    llavalcs=dict(
        type='InstructDataset',
        text_path=r"/data/Aaronzhu/DatasetStage1/Shikra/blip_laion_cc_sbu_558k_filter.jsonl",
        image_folder=r'/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/images',  # TODO: zz make folder name mistake
    ),
    instruct=dict(
        type='InstructDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/llava_instruct_150k.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/llava-instruct/images',
        add_coco_prefix=True,
    ),
    **train_gqa_variant,
    **train_clevr_variant,
    **train_point_variant,
    **train_gptgen_variant,
    **train_vcr_variant,
    **train_vqav2_variant,
    **train_vqaex_variant,
)

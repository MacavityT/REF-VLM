from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)



grit_train_common_cfg = dict(
    type='GRITDataset',
    text_path=r'/data/Aaronzhu/DatasetStage2and3/GRIT/annotations',
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/GRIT/img',
    stage=2,
    image_info_folder= None,
    offline_processed_image_folder = '',
)

train_grit_variant = dict(
    grit_c=dict(
        **grit_train_common_cfg, 
        version='c', 
        template_name=r"image_cap",
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grit_d=dict(
        **grit_train_common_cfg, 
        version='d', 
        template_name=r"DET",
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grit_cond_d=dict(
        **grit_train_common_cfg, 
        version='cond_d', 
        template_name=r"Cond_DET",
        placeholders=(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),
        offline_processed_text_folder='',        
    ),
    grit_r=dict(
        **grit_train_common_cfg, 
        version='r', 
        template_name=r"REC",
        placeholders=(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),
        offline_processed_text_folder='',
    ),
    grit_c_d=dict(
        **grit_train_common_cfg, 
        version='c_d', 
        template_name=r"flickr30k",
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grit_combine = dict(
        **grit_train_common_cfg, 
        version='combine', 
        template_name=["image_cap","DET","Cond_DET","REC","flickr30k"],
        placeholders=[(IMAGE_PLACEHOLDER,),(IMAGE_PLACEHOLDER,),(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),(IMAGE_PLACEHOLDER,)],
        offline_processed_text_folder='',
    )
)
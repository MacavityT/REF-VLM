from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    OBJS_PLACEHOLDER,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)



grit_train_common_cfg = dict(
    type='GRITDataset',
    text_path=r'/data/Aaronzhu/DatasetStage2and3/GRIT/annotations',
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/GRIT/img',
    stage=2,
)

train_grit_variant = dict(
    grit_c=dict(
        **grit_train_common_cfg, 
        version='c', 
        template_name=r"image_cap",
        map_placeholders=None,
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_caption_new',
    ),
    grit_d=dict(
        **grit_train_common_cfg, 
        version='d', 
        template_name=r"DET",
        map_placeholders=dict(
            output=["<boxes>"],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_det_new',
    ),
    grit_cond_d=dict(
        **grit_train_common_cfg, 
        version='cond_d', 
        template_name=r"Cond_DET",
        max_conv_length=5,
        map_placeholders=dict(
            output=["<boxes>"],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_cond_det_new',        
    ),
    grit_r=dict(
        **grit_train_common_cfg, 
        version='r', 
        template_name=r"REC",
        map_placeholders=dict(
            output=["<boxes>"],
        ),  
        placeholders=(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_rec',
    ),
    grit_g=dict(
        **grit_train_common_cfg, 
        version='g', 
        template_name=r"REG",
        map_placeholders=dict(
            input=["<boxes>"],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,OBJS_PLACEHOLDER),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_reg_noref',
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_reg',
    ),
    grit_c_d=dict(
        **grit_train_common_cfg, 
        version='c_d', 
        template_name=r"flickr30k",
        map_placeholders=dict(
            output=["<boxes>"],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/offline_single/offline_gcg',
    ),
    grit_combine = dict(
        **grit_train_common_cfg, 
        version='combine',
        max_conv_length=2,
        template_name=["image_cap","DET","Cond_DET","REC","REG","flickr30k"],
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),         
        placeholders=[(IMAGE_PLACEHOLDER,),(IMAGE_PLACEHOLDER,),(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),(IMAGE_PLACEHOLDER,OBJS_PLACEHOLDER),(IMAGE_PLACEHOLDER,)],
        offline_processed_text_folder='',
    ),
    grit_combine_offline = dict(
        type='GRITOfflineDataset',
        text_path=r'/data/Aaronzhu/DatasetStage2and3/GRIT/grit.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/GRIT/img',
        stage=2,
        version='combine_off',
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),         
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/GRIT/grit_offline',
    ),
)
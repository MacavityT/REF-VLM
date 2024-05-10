from xtuner.utils.constants import (
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASK_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)



grand_train_common_cfg = dict(
    type='GranDDataset',
    text_path=r'/data/Aaronzhu/GranD/GranD/Jsons',
    image_folder=r'/data/Aaronzhu/SA-1B/OpenDataLab___SA-1B/images',
    stage=2,
    image_info_folder= None,
    offline_processed_image_folder = '',
)

train_grand_variant = dict(
    grand_c=dict(
        **grand_train_common_cfg, 
        version='c', 
        use_floating_objects=True,
        template_name=r"image_cap",
        map_placeholders=None,
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grand_d=dict(
        **grand_train_common_cfg, 
        version='d', 
        use_floating_objects=True,
        template_name=r"DET",
        map_placeholders=dict(
            output=[BOXES_PLACEHOLDER],
        ),         
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grand_s=dict(
        **grand_train_common_cfg, 
        version='s', 
        use_floating_objects=True,
        template_name=r"SEG",
        map_placeholders=dict(
            output=[MASKS_PLACEHOLDER],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grand_cond_d=dict(
        **grand_train_common_cfg, 
        version='cond_d', 
        use_floating_objects=True,
        template_name=r"Cond_DET",
        map_placeholders=dict(
            output=[BOXES_PLACEHOLDER],
        ),     
        placeholders=(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),
        offline_processed_text_folder='',        
    ),
    grand_cond_s=dict(
        **grand_train_common_cfg,
        version='cond_s', 
        use_floating_objects=True,
        template_name=r"Cond_SEG",
        map_placeholders=dict(
            output=[MASKS_PLACEHOLDER],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),
        offline_processed_text_folder='',        
    ),
    grand_r_det=dict(
        **grand_train_common_cfg, 
        version='r_det', 
        use_floating_objects=True,
        template_name=r"REC",
        map_placeholders=dict(
            output=[BOXES_PLACEHOLDER],
        ),     
        placeholders=(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),
        offline_processed_text_folder='',
    ),
    grand_r_seg=dict(
        **grand_train_common_cfg, 
        version='r_seg', 
        use_floating_objects=True,
        template_name=r"RES",
        map_placeholders=dict(
            output=[MASKS_PLACEHOLDER],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),
        offline_processed_text_folder='',
    ),
    grand_re_det=dict(
        **grand_train_common_cfg, 
        version='re_det', 
        use_floating_objects=True,
        template_name=r"REG",
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
        ),     
        placeholders=(IMAGE_PLACEHOLDER,OBJS_PLACEHOLDER),
        offline_processed_text_folder='',
    ),
    grand_re_seg=dict(
        **grand_train_common_cfg, 
        version='re_seg', 
        use_floating_objects=True,
        template_name=r"REG_SEG",
        map_placeholders=dict(
            input=[MASKS_PLACEHOLDER],
        ), 
        placeholders=(IMAGE_PLACEHOLDER,MASK_PLACEHOLDER),
        offline_processed_text_folder='',
    ),

    grand_c_d=dict(
        **grand_train_common_cfg, 
        version='c_d', 
        use_floating_objects=True,
        template_name=r"flickr30k",
        map_placeholders=dict(
            output=[BOXES_PLACEHOLDER],
        ),     
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grand_c_s=dict(
        **grand_train_common_cfg, 
        version='c_s', 
        use_floating_objects=True,
        template_name=r"flickr30k_SEG",
        map_placeholders=dict(
            output=[MASKS_PLACEHOLDER],
        ),     
        placeholders=(IMAGE_PLACEHOLDER,),
        offline_processed_text_folder='',
    ),
    grand_mix = dict(
        **grand_train_common_cfg, 
        version='mix', 
        use_floating_objects=True,
        max_conv_length=2,
        template_name=["image_cap","DET","SEG","Cond_DET","Cond_SEG","REC","RES","REG","REG_SEG","flickr30k","flickr30k_SEG"],
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER,MASKS_PLACEHOLDER],
            output=[BOXES_PLACEHOLDER,MASKS_PLACEHOLDER],
        ),             
        placeholders=[(IMAGE_PLACEHOLDER,),
                      (IMAGE_PLACEHOLDER,),(IMAGE_PLACEHOLDER,),
                      (IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),(IMAGE_PLACEHOLDER,CLASS_PLACEHOLDER),
                      (IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),(IMAGE_PLACEHOLDER,EXPR_PLACEHOLDER),
                      (IMAGE_PLACEHOLDER,OBJS_PLACEHOLDER),(IMAGE_PLACEHOLDER,MASK_PLACEHOLDER),
                      (IMAGE_PLACEHOLDER,),(IMAGE_PLACEHOLDER,)],
        offline_processed_text_folder='',
    )
)
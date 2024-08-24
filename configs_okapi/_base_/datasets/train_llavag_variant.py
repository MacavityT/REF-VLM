llavag_train_common_cfg=dict(
    type='LLAVAGrounding', 
)


train_llavag_variant = dict(
    llavag_reg=dict(
        **llavag_train_common_cfg,
        version='reg',
        map_placeholders=dict(
            input=["<masks>"],
        ),
        text_path=r'/data/Aaronzhu/DatasetStage2and3/llava-grounding/llava_instruct_150k_visual_prompt.json',
        anno_path=r'/data/Aaronzhu/DatasetStage2and3/llava-grounding/coco/annotations/instances_train2014_filter.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/llava-grounding/offline_reg_new'     
    ),
    llavag_gcg=dict(
        **llavag_train_common_cfg,
        version='gcg',
        map_placeholders=dict(
            output=["<masks>"],
        ),
        text_path=r'/data/Aaronzhu/DatasetStage2and3/llava-grounding/coco/annotations/grounded_visual_chat_data.json',
        anno_path=r'/data/Aaronzhu/DatasetStage2and3/llava-grounding/coco/annotations/instances_train2017_gvc.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/llava-grounding/offline_gcg'           
    ),

)
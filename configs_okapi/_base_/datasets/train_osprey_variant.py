train_osprey_variant = dict(
    osprey_partlevel=dict(
        type='OspreyPartLevel', #2017
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_part_level.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_train_shape.jsonl',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/Osprey/offline/part',
        map_placeholders=dict(
            input=["<masks>"],
        )
    ),
    osprey_shortform=dict(
        type='OspreyShortForm', #2017
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_short_form.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_train_shape.jsonl',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/Osprey/offline/short_form',
        map_placeholders=dict(
            input=["<masks>"],
        )
    ),
    osprey_lvis=dict(
        type='OspreyLVISPosNeg', #2017
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_lvis_positive_negative.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_train_shape.jsonl',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/Osprey/offline/lvis',
        map_placeholders=dict(
            input=["<masks>"],
        )
    ),
    osprey_conversations=dict(
        type='OspreyConversations', #2014
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_conversation.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2014_train_shape.jsonl',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/Osprey/offline/conversation',
        map_placeholders=dict(
            input=["<masks>"],
        )
    ),
    osprey_detailed=dict(
        type='OspreyDetailedDescription', #2014
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_detail_description.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2014_train_shape.jsonl',
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/Osprey/offline/detail',
        map_placeholders=dict(
            input=["<masks>"],
        )
    )
)
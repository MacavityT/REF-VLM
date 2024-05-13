from xtuner.utils.constants import MASKS_PLACEHOLDER

train_osprey_variant = dict(
    ospery_partlevel=dict(
        type='OspreyPartLevel', #2017
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_part_level.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_train_shape.jsonl',
        map_placeholders=dict(
            input=[MASKS_PLACEHOLDER],
        )
    ),
    ospery_shortform=dict(
        type='OspreyShortForm', #2017
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_short_form.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_train_shape.jsonl',
        map_placeholders=dict(
            input=[MASKS_PLACEHOLDER],
        )
    ),
    ospery_lvis=dict(
        type='OspreyLVISPosNeg', #2017
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_lvis_positive_negative.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_train_shape.jsonl',
        map_placeholders=dict(
            input=[MASKS_PLACEHOLDER],
        )
    ),
    ospery_conversations=dict(
        type='OspreyConversations', #2014
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_conversation.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2014_train_shape.jsonl',
        map_placeholders=dict(
            input=[MASKS_PLACEHOLDER],
        )
    ),
    ospery_detailed=dict(
        type='OspreyDetailedDescription', #2014
        text_path=r'/data/Aaronzhu/DatasetStage2and3/Osprey/Osprey-724K/osprey_detail_description.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2014_train_shape.jsonl',
        map_placeholders=dict(
            input=[MASKS_PLACEHOLDER],
        )
    )
)
import json
import json
import os
import random
import numpy as np
from collections import namedtuple
from typing import Dict, List
from PIL import Image
from vt_plug.registry import DATASETS
from collections import defaultdict
import pycocotools.mask as mask_utils
from pycocotools.mask import decode,encode
from tqdm import tqdm
import pickle
from vt_plug.utils.constants import (
    IMAGE_PLACEHOLDER,
    QUESTION_PLACEHOLDER, 
    OBJS_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    EXPR_PLACEHOLDER,
    CLASS_PLACEHOLDER
)
from .mixin import MInstrDataset
from mmengine.fileio import get
from panopticapi import utils
from mmdet.datasets.api_wrappers.coco_api import COCOPanoptic
from vt_plug.registry import DATASETS
from vt_plug.utils.constants import IMAGE_PLACEHOLDER
from vt_plug.dataset.utils import imfrombytes

@DATASETS.register_module()
class PNGDataset(MInstrDataset):
    def __init__(self, *args, version, anno_path,anno_img_dir, **kwargs):
        self.version = version
        self.anno_path = anno_path
        self.anno_img_dir = anno_img_dir
        assert self.version in ['reg_mask','reg_box','gcg']
        super().__init__(*args, **kwargs)
        self.coco = COCOPanoptic(self.anno_path)


    @staticmethod
    def _load_segm(segm_path):
        pan_png = imfrombytes(segm_path, flag='color', channel_order='rgb').squeeze()
        segm_map = utils.rgb2id(pan_png)

        return segm_map

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item
        item = self.text_data[index]
        image_id = int(item['image_id'])
        annotations = {ann['id']: ann for ann in self.coco.imgToAnns[image_id]}
        image_info_dict = self.coco.imgs[image_id]
        image_info = {'path': os.path.join(self.image_folder,image_info_dict['file_name']),
                      'height': image_info_dict['height'],
                      'width': image_info_dict['width']}
        segm_file = image_info_dict['segm_file']
        segm_map = self._load_segm(os.path.join(self.anno_img_dir, segm_file))
        caption_segments = item['segments']
        gt_masks = []
        if self.version == 'gcg':
            new_caption = ''
            masks_seq = []
            for i, segment in enumerate(caption_segments):
                caption = segment['utterance']
                if segment['segment_ids'] != []:
                    mask_seq = []
                    for segment_id in segment['segment_ids']:
                        segment_mask = (segm_map == int(segment_id)).astype(np.uint8)
                        gt_masks.append(segment_mask)
                        mask_seq.append(len(gt_masks)-1)
                    caption = PHRASE_ST_PLACEHOLDER_STAGE2 + caption + PHRASE_ED_PLACEHOLDER_STAGE2 + MASKS_PLACEHOLDER * len(segment['segment_ids'])
                    masks_seq.append(mask_seq)
                new_caption += ' ' + caption
            if masks_seq == []:
                question = "Please describe the content of the image<image> in a few words."
                ret = {
                    'image':image_info,
                    'conversations': [
                        {'from':'system','value':[{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]},
                        {'from': 'human','value': question},
                        {'from': 'gpt','value': item['caption']}
                    ]
                }
                return ret
            question = self.get_template()
            new_caption = new_caption.replace(' ,',',').replace(' .','.').strip()

            
            ret = {
                'image':image_info,
                'target':{'masks':gt_masks},
                'conversations': [
                    {'from':'system','value':[{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']}]},
                    {'from': 'human','value': question},
                    {'from': 'gpt','value': new_caption,'masks_seq':masks_seq},
                ]
            }
            ret['map_placeholders'] = self.map_placeholders

        
        
#         # TODO: add reg tasks
#         # elif self.version == 'reg_mask':
            
#             # all_conversations = []
#             # for i, segment in enumerate(caption_segments):
#             #     caption = segment['utterance']
#             #     if segment['segment_ids'] != []:
#             #         for segment_id in segment['segment_ids']:
#             #             question = self.get_template()
#             #             question = question.replace()
#             #             segment_mask = self.anns[int(segment_id)]['mask']
#             #             gt_masks.append(segment_mask)

#             #             human = {'from':'human','value':question}
#             #             all_conversations.append()
        
        return ret

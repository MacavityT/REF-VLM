import json
from vt_plug.registry import DATASETS
import os
from vt_plug.utils.constants import IMAGE_PLACEHOLDER,DEPTH_PLACEHOLDER
from PIL import Image
import numpy as np
import re
from .mixin import MInstrDataset


@DATASETS.register_module()
class NYUDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert os.path.isdir(self.text_path), "NYU dataset is composed of list of gt_depth files, not a single json!"
        self.img_depth_pair = []
        for folder in os.listdir(self.text_path):
            gt_path = os.path.join(self.text_path,folder)
            for img_path in os.listdir(gt_path):
                if img_path[-3:] == 'png':
                    depth_path = os.path.join(gt_path,img_path)
                    number = re.search(r'\d+', img_path).group()
                    img_path = os.path.join(gt_path,f"rgb_{number}.jpg")


                    self.img_depth_pair.append({'depth_path':depth_path,'img_path':img_path})

    def __len__(self):
        if (self.offline_processed_text_folder is not None) and os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.img_depth_pair)
        
    def __getitem__(self, index):
        gt_depth_dict = self.img_depth_pair[index]
        gt_depth_file = gt_depth_dict['depth_path']
        image_path = gt_depth_dict['img_path']
        gt_depth = np.array(Image.open(gt_depth_file))
        question = self.get_template()


        ret = {
            'image':{'path':image_path,'width':gt_depth.shape[1],'height':gt_depth.shape[0]},
            'target':{'depth':[gt_depth]},
            'conversations':[
                {'from':'system',
                 'value':[{'task':{'task_name':'depth',
                                   'element':['sentence','phrase'],'use_unit':True},
                            'unit':['depth']}]},
                {
                    'from':'human',
                    'value':question
                    },
                {
                    'from':'gpt',
                    'value': f"Here is the depth map: <Phrase>target</Phrase>{DEPTH_PLACEHOLDER}.",
                    'depth_seq':[[0]],
                 }
            ]
        }

        ret['map_placeholders'] = self.map_placeholders

        return ret
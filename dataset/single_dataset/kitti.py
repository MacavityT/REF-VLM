import json
from xtuner.registry import DATASETS
import os
from xtuner.utils.constants import IMAGE_PLACEHOLDER,DEPTH_PLACEHOLDER
from PIL import Image
import numpy as np
from .mixin import MInstrDataset


@DATASETS.register_module()
class KITTIDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert os.path.isdir(self.text_path), "KITTI dataset is composed of list of gt_depth files, not a single json!"
        self.gt_depth = []

        for folder in os.listdir(self.text_path):
            gt_path = os.path.join(self.text_path,folder,'proj_depth','groundtruth')
            for img_path in os.listdir(gt_path):
                depth_path = os.path.join(gt_path,img_path)
                for file_name in os.listdir(depth_path):
                    gt_depth = os.path.join(depth_path,file_name)

                    gt_depth_dict = {
                        'image_folder':folder,
                        'image_index':img_path,
                        'image_name':file_name,
                        'depth_path':gt_depth,
                    }
                    self.gt_depth.append(gt_depth_dict)

    def __len__(self):
        if (self.offline_processed_text_folder is not None) and os.path.exists(self.offline_processed_text_folder):
            return len(self.text_data)
        else:
            return len(self.gt_depth)
        
    def __getitem__(self, index):
        gt_depth_dict = self.gt_depth[index]
        gt_depth_file = gt_depth_dict['depth_path']

        image_time = gt_depth_dict['image_folder'].split('_drive')[0]
        image_path = os.path.join(self.image_folder,image_time,
                                  gt_depth_dict['image_folder'],
                                  gt_depth_dict['image_index'],'data',
                                  gt_depth_dict['image_name'])

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
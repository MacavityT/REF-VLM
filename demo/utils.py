import os
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import torch
from torch.utils.data import Dataset
from ref_vlm.utils.constants import IMAGE_PLACEHOLDER

class ImageBoxState:
    def __init__(self, draw_size=512):
        if isinstance(draw_size, (float, int)):
            draw_size = (draw_size, draw_size)
        assert len(draw_size) == 2
        self.size = draw_size
        self.height, self.width = self.size[0], self.size[1]
        self.reset_state()
        self.cnt = 0

    # noinspection PyAttributeOutsideInit
    def reset_state(self):
        self.image = None
        self.boxes = []
        self.masks = []

    # noinspection PyAttributeOutsideInit
    def reset_masks(self):
        self.boxes = []
        self.masks = []

    # noinspection PyAttributeOutsideInit
    def update_image(self, image):
        if image != self.image:
            # self.reset_state()
            self.image = image

    def update_mask(self, mask):
        if len(self.masks) == 0:
            last_mask = np.zeros_like(mask)
        else:
            last_mask = self.masks[-1]

        if type(mask) == np.ndarray and mask.size > 1:
            diff_mask = mask - last_mask
        else:
            diff_mask = np.zeros([])

        # clear all of the strokes
        if mask.sum() == 0:
            self.reset_masks()
            return

        if (mask.astype(np.float32) - last_mask.astype(np.float32)).sum()<0:
            self.boxes.pop()
            self.masks.pop()
            return

        if diff_mask.sum() > 0:
            # noinspection PyArgumentList
            x1x2 = np.where(diff_mask.max(0) != 0)[0]
            # noinspection PyArgumentList
            y1y2 = np.where(diff_mask.max(1) != 0)[0]
            y1, y2 = y1y2.min(), y1y2.max()
            x1, x2 = x1x2.min(), x1x2.max()
            if (x2 - x1 > 5) and (y2 - y1 > 5):
                self.masks.append(mask.copy())
                self.boxes.append(tuple(map(int, (x1, y1, x2, y2))))

    def update_box(self, box):
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.boxes.append(tuple(map(int, (x1, y1, x2, y2))))

    def to_model(self):
        pass
        # if self.image is None:
        #     return {}
        # image = expand2square(self.image)
        # boxes = [box_xyxy_expand2square(box, w=self.image.width, h=self.image.height) for box in self.boxes]
        # return {'image': image, 'boxes': boxes}

    def draw_boxes(self):
        assert self.image is not None
        grounding_texts = [f'{bid}' for bid in range(len(self.boxes))]
        def _draw(img, _boxes, texts):
            assert img is not None
            colors = ["red", "blue", "green", "olive", "orange", "brown", "cyan", "purple"]
            _img_draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'assets/DejaVuSansMono.ttf'), size=18)
            for bid, box in enumerate(_boxes):
                _img_draw.rectangle((box[0], box[1], box[2], box[3]), outline=colors[bid % len(colors)], width=4)
                anno_text = texts[bid]
                _img_draw.rectangle((box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]),
                                    outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
                _img_draw.text((box[0] + int(font.size * 0.2), box[3] - int(font.size * 1.2)), anno_text, font=font, fill=(255, 255, 255))
            return img

        out_draw = _draw(self.image, self.boxes, grounding_texts)
        return out_draw


def bbox_draw(sketch_pad: dict, state: dict):
    def binarize(x):
        return (x != 0).astype('uint8') * 255
    image = sketch_pad['image']
    image = open_image(image)
    # global count
    # count += 1
    # np.save( f"{count}.npy", sketch_pad['mask'])
    mask = sketch_pad['mask'].sum(-1) if sketch_pad['mask'].ndim == 3 else sketch_pad['mask']
    mask = binarize(mask)
    ibs = state["ibs"]
    ibs.update_image(image)
    ibs.update_mask(mask)
    out_draw = ibs.draw_boxes()
    return out_draw, state

MAP_PLACEHOLDER = {
    'vqa'
}

class SingleInferDataset(Dataset):
    def __init__(self):
        self.rets = []
        self.map_placeholders = dict(
            input=["<boxes>","<masks>"],
            output=["<boxes>","<masks>"],
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conversations = []
        self.image = None
        self.system = {'from':'system','value':[]}
        self.target = {}


    def add_image(self,image):
        self.image = {'value':image}

    def append_message_question(self,input_text,system_value,vpt_mask):
        if self.image is not None:
            self.input_text = {'from': 'human', 'value': IMAGE_PLACEHOLDER + input_text}
        else:
            self.input_text = {'from': 'human', 'value': input_text}
        if vpt_mask != []:
            self.input_text['masks_seq'] = [[0]]
            self.target['masks'] = vpt_mask
        self.conversations.append(self.input_text)
        self.conversations.append({'from': 'gpt', 'value': ''})
        self.system['value'].append(system_value)

    def add_one_conversation(self):
        ret = {}
        ret['map_placeholders'] = self.map_placeholders
        if self.image is None:
            ret['conversations'] = self.conversations
        else:
            assert len(self.system['value']) == len(self.conversations) // 2
            self.conversations.insert(0,self.system)
            ret['image'] = self.image
            if self.target != {}:
                ret['target'] = self.target
            ret['conversations'] = self.conversations
            
        self.rets = [ret]
        self.reset_parameters()

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        assert self.rets != []
        return self.rets[index]
        

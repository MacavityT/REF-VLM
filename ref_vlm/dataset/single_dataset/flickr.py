import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import json
from collections import defaultdict
from pycocotools.mask import decode
from torch.utils.data import Dataset
from ref_vlm.registry import DATASETS
from ref_vlm.utils.constants import (
    BOXES_PLACEHOLDER, 
    IMAGE_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2
    )
from .mixin import MInstrDataset

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r', encoding='utf8') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index': index,
                                             'phrase': phrase,
                                             'phrase_id': p_id,
                                             'phrase_type': p_type})

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def get_ann_path(idx, *, annotation_dir=""):
    return os.path.join(annotation_dir, rf'Annotations/{idx}.xml')


def get_sen_path(idx, *, annotation_dir=""):
    return os.path.join(annotation_dir, rf"Sentences/{idx}.txt")


def get_img_path(idx, *, image_dir=""):
    return os.path.join(image_dir, rf'{idx}.jpg')


def flatten_annotation(annotation_dir, indexes):
    data = []

    for index in tqdm(indexes):
        image_id = index
        ann_path = get_ann_path(index, annotation_dir=annotation_dir)
        sen_path = get_sen_path(index, annotation_dir=annotation_dir)
        anns = get_annotations(ann_path)
        sens = get_sentence_data(sen_path)

        for sen in sens:
            pids = list(set(phrase['phrase_id'] for phrase in sen['phrases'] if phrase['phrase_id'] in anns['boxes']))
            boxes_mapping: Dict[str, List[int]] = {}
            boxes_filtered: List[List[int]] = []
            for pid in pids:
                v = anns['boxes'][pid]
                mapping = []
                for box in v:
                    mapping.append(len(boxes_filtered))
                    boxes_filtered.append(box)
                boxes_mapping[pid] = mapping

            boxes_seq: List[List[int]] = []
            for phrase in sen['phrases']:
                if not phrase['phrase_id'] in anns['boxes']:
                    continue
                pid = phrase['phrase_id']
                boxes_seq.append(boxes_mapping[pid])

            sent = list(sen['sentence'].split())
            for phrase in sen['phrases'][::-1]:
                if not phrase['phrase_id'] in anns['boxes']:
                    continue
                span = [phrase['first_word_index'], phrase['first_word_index'] + len(phrase['phrase'].split())]
                sent[span[0]:span[1]] = [f"{PHRASE_ST_PLACEHOLDER}{' '.join(sent[span[0]:span[1]])}{PHRASE_ED_PLACEHOLDER}"]
            sent_converted = " ".join(sent)

            assert len(re.findall(PHRASE_ST_PLACEHOLDER, sent_converted)) \
                   == len(re.findall(PHRASE_ED_PLACEHOLDER, sent_converted)) \
                   == len(boxes_seq), f"error when parse: {sent_converted}, {boxes_seq}, {sen}, {anns}"
            assert sent_converted.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, "") == sen['sentence']

            item = {
                'id': len(data),
                'image_id': image_id,
                'boxes': boxes_filtered,
                'sentence': sent_converted,
                'boxes_seq': boxes_seq,
            }
            data.append(item)

    return data


# process flickr dataset online
class FlickrParser(Dataset):
    def __init__(self, filename, annotation_dir):
        self.filename = filename
        self.annotation_dir = annotation_dir

        self.indexes = [line.strip() for line in open(filename, 'r', encoding='utf8')]
        self.data = flatten_annotation(self.annotation_dir, self.indexes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def dump(self, filename):
        import json
        with open(filename, 'w', encoding='utf8') as f:
            for obj in self.data:
                obj_str = json.dumps(obj)
                f.write(obj_str)
                f.write('\n')


@DATASETS.register_module()
class FlickrDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item

        item = self.get_raw_item(index)
        img_path = f"{item['image_id']}.jpg"
        caption = item['sentence']

        image = self.get_image(img_path)
        question = self.get_template()

        if self.stage == 1:
            caption = caption.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            ret = {
                'image': image,
                'target': {'boxes': item['boxes']},  # 'seg' /
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                        'boxes_seq': item['boxes_seq'],
                    }
                ]
            }

        if self.stage == 2:
            caption = caption.replace(PHRASE_ED_PLACEHOLDER,f"{PHRASE_ED_PLACEHOLDER}{BOXES_PLACEHOLDER}")
            caption = caption.replace(PHRASE_ST_PLACEHOLDER,PHRASE_ST_PLACEHOLDER_STAGE2).replace(PHRASE_ED_PLACEHOLDER,PHRASE_ED_PLACEHOLDER_STAGE2)
            boxes_list = [BOXES_PLACEHOLDER * len(box_seq) for box_seq in item['boxes_seq']]
            caption = caption.replace(BOXES_PLACEHOLDER,'{}').format(*boxes_list)

            value = [{'task':{'task_name':'gcg_detection','element':['phrase','sentence'],'use_unit':True},'unit':['box']}]



            ret = {
                'image': image,
                'target': {'boxes': item['boxes']},  # 'seg' /
                'conversations': [
                    {
                        'from': 'system',
                        'value':value,
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                        'boxes_seq': item['boxes_seq'],
                    }
                ]
            }

            if item['boxes_seq'] == []:
                ret['conversations'][0]['value'] = [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}]
                ret['conversations'][1]['value'] = "Can you give me a brief description of this image<image>?"
                del ret['conversations'][2]['boxes_seq'] 

            ret['map_placeholders'] = self.map_placeholders
        return ret



@DATASETS.register_module()
class FlickrCaptionDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item

        item = self.get_raw_item(index)
        img_path = f"{item['image_id']}.jpg"
        caption = item['sentence']

        image = self.get_image(img_path)
        question = self.get_template()

        if self.stage == 1:
            caption = caption.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, "")
            caption = re.sub(r'\s+(,|\.)', r'\1', caption)
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                    }
                ]
            }

        if self.stage == 2:
            caption = caption.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, "")
            caption = re.sub(r'\s+(,|\.)', r'\1', caption)
            ret = {
                'image': image,
                'conversations': [
                    {
                        'from': 'system',
                        'value': [{'task':{'task_name':'vqa','element':['sentence'],'use_unit':False}}],
                    },
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': caption,
                    }
                ]
            }
            ret['map_placeholders'] = self.map_placeholders
        return ret



@DATASETS.register_module()
class FlickrSegmentationDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        offline_item = super().__getitem__(index)
        if offline_item is not None:
            return offline_item

        item = self.get_raw_item(index)
        img_path = f"{item['image_id']}.jpg"
        caption = item['sentence']

        image = self.get_image(img_path)
        question = self.get_template()
        value = [{'task':{'task_name':'gcg_segmentation','element':['phrase','sentence'],'use_unit':True},'unit':['mask']}]

        masks = []
        for mask in item['masks']:
            masks.append(decode(mask))

        ret = {
            'image': image,
            'target': {'masks': masks},  # 'seg' /
            'conversations': [
                {
                    'from': 'system',
                    'value':value,
                },
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                    'masks_seq': item['masks_seq'],
                }
            ]
        }
        ret['map_placeholders'] = self.map_placeholders

        return ret

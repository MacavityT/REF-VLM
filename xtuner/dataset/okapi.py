import warnings
from functools import partial
from typing import Dict, Any, Callable, List, Optional, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments

from xtuner.utils.constants import IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER
from ..conversation import Conversation, get_conv_template
from ..utils import post_process_generate_ids


class SingleImageConvDatasetMixin:

    def __init__(
            self,
            *args,
            preprocessor: Dict[str, Any],
            process_func: Dict[str, Any],
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='vicuna_v1.1'),
            mode='train',
            tokenize_kwargs: dict = None,
            training_args: TrainingArguments = None,
            transforms: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode in ['train', 'validation', 'test']

        self.preprocessor = preprocessor
        self.process_func = process_func
        self.conv_template = conv_template
        self.mode = mode
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.training_args = training_args
        self.transforms = transforms

    def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
        # getitem
        item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # transform
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            has_image = 'image' in item and bool(item['image'])
            has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
            if has_target and has_image:
                target['width'], target['height'] = image.width, image.height
        
        #TODO: 改为遍历 process config ，自动build并处理
        # preprocess
        raw_conv = self.process_conv(raw_conv)
        raw_conv, image = self.process_conv_multimage(raw_conv, image)
        raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
        conv = self.build_conv(raw_conv)
        if return_conv:
            # noinspection PyTypeChecker
            return conv
        text_dict = self.process_text(conv)
        image_dict = self.process_image(image)

        # return
        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)
        self._print_sample(ret_dict, raw_conv, conv)
        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
        return ret_dict

    def __len__(self):
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def process_conv_multimage(self, raw_conv, image):
        # re-sort multi image
        if image is None:
            return raw_conv, image
        if not isinstance(image, (list, tuple)):
            return raw_conv, image
        image_seqs = []
        for conv in raw_conv:
            image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
        images = []
        for idx in image_seqs:
            images.append(image[idx])
        return raw_conv, images

    def get_raw_item(self, index) -> Dict[str, Any]:
        """
        return item format like this.
        item = {
            'image': # PIL.Image.Image,
            'target': {
                # xmin, ymin, xmax, ymax
                'boxes': [
                    [10, 10, 256, 265],  # dog1
                    [24, 18, 378, 768],  # dog2
                    [100, 310, 670, 653],  # man
                    [278, 320, 809, 673],  # rope
                ],
            }

            "conversations": [
                {
                    'from': 'human',
                    'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
                    'boxes_seq': [[0, 1], [2], ],
                },
                {
                    'from': 'gpt',
                    'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                             'So the man <boxes> is walking the dog <boxes>.'
                            'And the man <boxes> has no relationship with the right dog <boxes>',
                    'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
                }
            ]
        }
        # placeholder: <image> <boxes>
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def validate_raw_item(self, item):
        has_image = 'image' in item and bool(item['image'])
        has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
        has_target_boxes = 'boxes' in item['target'] if has_target else False
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # check image
        human_input_has_image_placeholder = any(
            sentence['from'] == 'human' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if human_input_has_image_placeholder:
            assert has_image
        if has_image and (not human_input_has_image_placeholder):
            warnings.warn(f'item has image but the question has no image placeholder.\n{item}')
        gpt_input_has_image_placeholder = any(
            sentence['from'] == 'gpt' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        assert not gpt_input_has_image_placeholder

        # check target
        has_boxes_placeholder = any(
            BOXES_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if has_boxes_placeholder:
            assert has_target_boxes
        # not check box placeholder num this will be checked in format process

    def build_conv(self, source: List[Dict[str, Any]]) -> Conversation:
        conv = self.conv_template()
        role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
        assert len(source) > 0
        assert source[0]['from'] == 'human'
        for sentence in source:
            role = role_map[sentence['from']]
            conv.append_message(role, sentence['value'])
        return conv

    def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
        return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)

    def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], multimage_mode=False) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
        return self.process_func['target'](raw_conv, target, self.preprocessor, multimage_mode=multimage_mode)

    def process_text(self, conv: Conversation) -> Dict[str, Any]:
        """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
        return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        convert Image.Image object to torch.Tensor
        """
        return self.process_func['image'](image, self.preprocessor)

    def _print_sample(self, ret_dict, raw_conv, conv):
        if not hasattr(self, '_printed_sample'):
            self._printed_sample = True
            post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
            print(f"=================== {self.mode} sample ===================", flush=True)
            print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
            print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
            print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
            print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
            if 'image' in ret_dict and ret_dict['image'] is not None:
                image = ret_dict['image']
                if isinstance(image, torch.Tensor):
                    print(f"            image: {image.shape}")
                elif isinstance(image, dict):
                    print(f"            image: {image.keys()}")
                elif isinstance(image, list) and len(image) > 0:
                    print(f"            image: {len(image)}, {type(image[0])}")
                else:
                    print(f"            image: {type(image)}")
            print("====================================================", flush=True)
            try:
                if self.training_args is not None:
                    _save_obj = {
                        'ret_dict': ret_dict,
                        'raw_conv': raw_conv,
                        'conv': conv.get_prompt(),
                    }
                    from pathlib import Path
                    output_dir = Path(self.training_args.output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    _local_rank = self.training_args.local_rank
                    _word_size = self.training_args.world_size
                    _file_path = str(output_dir / f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
                    print(f'saving some sample to {_file_path} for check.')
                    torch.save(_save_obj, _file_path)
            except Exception as e:
                warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')


class SingleImageConvDataset(SingleImageConvDatasetMixin, Dataset):
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None

    def initialize_if_needed(self):
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            # warnings.warn("it's highly recommended that set persistent_workers=True, "
            #               "otherwise this initialize code will run in every epoch beginning."
            #               "(ignore me if set)")
            self.dataset = self.dataset_generator()

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


__all__ = ['SingleImageConvDatasetMixin', 'SingleImageConvDataset']


# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from datetime import timedelta
from functools import partial

import numpy as np
from datasets import DatasetDict, concatenate_datasets
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from torch import distributed as dist

from xtuner.registry import BUILDER, MAP_FUNC
from .utils import Packer, encode_fn
from .huggingface import (
    get_lengths,
    build_origin_dataset,
    map_dataset,
    add_template_to_dataset,
    tokenize_dataset,
    pack_dataset,
    process
)

def process_single_dataset(dataset,
                       do_dataset_tokenization=True,
                       tokenizer=None,
                       max_length=None,
                       dataset_map_fn=None,
                       template_map_fn=None,
                       max_dataset_length=None,
                       split='train',
                       remove_unused_columns=False,
                       rename_maps=[],
                       shuffle_before_pack=True,
                       pack_to_max_length=True,
                       use_varlen_attn=False,
                       input_ids_with_output=True,
                       with_image_token=False,
                       map_num_proc=32):
    """Post-process the dataset loaded from the Hugging Face Hub, or a local
    dataset.

    Args:
        dataset: The dataset to be post-processed.
        do_dataset_tokenization: Whether the dataset need to be tokenized
            in this function. Default to True.
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding. If `do_dataset_tokenization` is True, this argument
            should not be None. Default to None.
        max_length: Max length of the sequence. If `do_dataset_tokenization`
            or `pack_to_max_length` is True, this argument should not be None.
            Default to None.
        dataset_map_fn: Map the original dataset format to the one defined
            by xTuner.
        template_map_fn: Add the prompt template to the dataset
        max_dataset_length: If the length of the dataset is too long, we can
            randomly extract `max_dataset_length` from it.
        split: Which split of the data to load.
            If `None`, will return a single concatenated dataset with all
            splits (typically `datasets.Split.TRAIN` and
            `datasets.Split.TEST`).
            If given, will return a single Dataset.
        remove_unused_columns: Whether to remove columns from the dataset
            that are not used during training.
        rename_maps: Rename the column name of the dataset.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        use_varlen_attn: If use_varlen_attn is True, we calculate attention
            the actual length of the sequence rather than the actual length
            of the sequence
        input_ids_with_output: Whether to put the groundtruth output
            corresponding to the question into the dataset. Typically set
            it to True during training and False during testing.
        with_image_token: Whether to convert DEFAULT_IMAGE_TOKEN to
            IMAGE_TOKEN_INDEX. Typically set it to True during the training
            of VLM.
        map_num_proc: Max number of processes when mapping the dataset.
    """
    kwargs = dict(
        dataset=dataset,
        do_dataset_tokenization=do_dataset_tokenization,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=dataset_map_fn,
        template_map_fn=template_map_fn,
        max_dataset_length=max_dataset_length,
        split=split,
        remove_unused_columns=remove_unused_columns,
        rename_maps=rename_maps,
        shuffle_before_pack=shuffle_before_pack,
        pack_to_max_length=pack_to_max_length,
        use_varlen_attn=use_varlen_attn,
        input_ids_with_output=input_ids_with_output,
        with_image_token=with_image_token,
        map_num_proc=map_num_proc)
    if not (dist.is_available() and dist.is_initialized()):
        return process(**kwargs)

    xtuner_dataset_timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=30)))
    print_log(
        f'xtuner_dataset_timeout = {xtuner_dataset_timeout}', logger='current')
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_dataset_timeout)

    if dist.get_rank() == 0:
        dataset = process(**kwargs)
        objects = [dataset]
    else:
        objects = [None]

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_dataset_timeout)
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os
import numpy as np

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER, DATASETS, FUNCTIONS
from .huggingface import process_hf_dataset
from .utils import expand2square

class OkapiDataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 offline_processed_image_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        assert offline_processed_image_folder or (image_folder and image_processor)
        if offline_processed_image_folder and image_folder:
            print_log(
                'Both `offline_processed_image_folder` and '
                '`image_folder` are set, and we load dataset from'
                '`offline_processed_image_folder` '
                f'({offline_processed_image_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            json_data = json.load(open(data_path))
            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True)

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

    def load_offline_image_data():
        pass

    def load_offline_text_data():
        pass

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict


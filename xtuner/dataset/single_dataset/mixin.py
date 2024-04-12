import json
import os
import logging

import numpy as np
from PIL import Image
from mmengine import print_log
from mmengine.config import Config, ConfigDict

from torch.utils.data import Dataset
from datasets import DatasetDict, load_from_disk
from xtuner.registry import BUILDER
from .dataset_templates import dataset_template_path

class QuestionTemplateMixin:
    def __init__(
            self,
            *args,
            template_name=None,
            template_file=None,
            max_dynamic_size=None,
            placeholders=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.template_file = template_file
        self.max_dynamic_size = max_dynamic_size
        self.placeholders = placeholders

        assert template_name or template_file, ("assign neither template_name nor template_file")
        if template_name is not None and template_file is not None:
            raise ValueError(f"assign both template_name and template_file:\nstring:{template_name}\nfile:{template_file}")
        if template_name is not None:
            self.template_file = dataset_template_path[template_name]

        self.templates = json.load(open(template_file, 'r', encoding='utf8'))
        if self.max_dynamic_size is not None:
            self.templates = self.templates[: self.max_dynamic_size]

        # sanity check
        assert self.placeholders is not None
        for template in self.templates:
            for placeholder in placeholders:
                assert str(template).count(placeholder) == 1, f"template: {template}\nplaceholder:{placeholder}"

    def get_template(self):
        import random
        return random.choice(self.templates)

    def template_nums(self):
        return len(self.templates)


class MInstrDataset(QuestionTemplateMixin, Dataset):
    _repr_indent = 4

    def __init__(self,
                text_path, 
                image_folder=None, 
                offline_processed_text_folder=None,
                offline_processed_image_folder=None,
                tokenizer=None,
                image_processor=None,
                pad_image_to_square=False,
                seed=None, 
                **kwargs):
        super().__init__(**kwargs)
        self.text_path = text_path
        self.image_folder = image_folder
        self.rng = np.random.default_rng(seed)

        self.tokenizer = tokenizer
        self.image_processror = image_processor,
        self.pad_image_to_square = pad_image_to_square
        self.text_data = []

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        assert offline_processed_text_folder or (text_path and tokenizer)
        if offline_processed_text_folder and text_path:
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
            self.text_data = self.load_offline_text_data(offline_processed_text_folder)
        if offline_processed_image_folder is not None:
            self.image_data = self.load_offline_image_data(offline_processed_image_folder)

    #TODO: 增加离线加载流程
    def load_offline_text_data(offline_processed_text_folder):
        return load_from_disk(offline_processed_text_folder)

    def load_offline_image_data(offline_processed_image_folder, image_path):
        raise NotImplementedError

    def get_raw_item(self, index):
        return json.loads(self.text_data[index])
    
    def get_text(self, text_path):
        with open(text_path, 'r', encoding='utf8') as f:
            # for line in tqdm(f, desc=f'{self.__class__.__name__} loading ann {self.filename}'):
            for line in f:
                self.text_data.append(line)

    def get_image(self, image_path):
        # if self.image_folder is not None:
        #     image_path = os.path.join(self.image_folder, image_path)
        # image = Image.open(image_path).convert('RGB')
        return image_path

    def get_template(self):
        return self.rng.choice(self.templates)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.text_data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
            f"ann file: {self.text_path}"
        ]
        if self.image_folder is not None:
            body.append(f"image folder: {self.image_folder}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    # noinspection PyMethodMayBeStatic
    def extra_repr(self) -> str:
        return ""

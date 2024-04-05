# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from xtuner.dataset.map_fns.dataset_templates import dataset_template_path

DATASET_TEMPLATES = dataset_template_path.keys()

#TODO: add each dataset map fn
def vqa_map_fn(example):
    print(example)
    pass

def okapi_map_fn(example, template: str):
    assert template in DATASET_TEMPLATES, f"{template} is not supported!"
    template_file = dataset_template_path[template]
    map_fn = getattr(__import__(__name__), template)

    


    # # origin llava map fn
    # messages = example['conversations']
    # input = ''
    # conversation = []
    # while messages and messages[0]['from'] == 'gpt':
    #     # Skip the first one if it is from gpt
    #     messages = messages[1:]
    # for msg in messages:
    #     if msg['from'] == 'human':
    #         if DEFAULT_IMAGE_TOKEN in msg['value']:
    #             msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
    #                                                 '').strip()
    #             msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
    #             msg['value'] = msg['value'].strip()
    #         input += msg['value']

    #     elif msg['from'] == 'gpt':
    #         conversation.append({'input': input, 'output': msg['value']})
    #         input = ''
    #     else:
    #         raise NotImplementedError
    # return {'conversation': conversation}

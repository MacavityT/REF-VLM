# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict

# - Turn 0: SYSTEM + INSTRUCTION, [output + SUFFIX], SEP
# - Turn 1: INSTRUCTION, [output + SUFFIX], SEP
# - Turn ...
# Note: [] means having supervised loss during the fine-tuning
PROMPT_TEMPLATE = ConfigDict(
    vicuna=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    vd_cot=dict(
        SYSTEM_PREFIX=('A chat between curious user and an artificial intelligence assistant ' 
                'capable of handling common computer vision tasks. '
                'The assistant provides an answer to the user\'s questions based on a specific task command.\n'
                'Capabilities and tools that assistant can possess:\n'
                '- Sentence: gives helpful, detailed, and polite answers.\n'
                '- Phrase: gives short, precise answers, follow the format of \'<Phrase>phrase</Phrase>\'.\n'
                '- Unit: gives answers with given unit name, follow the format of \'<Unit>unit name</Unit>[number]\'.\n'),
        SYSTEM=('{system}\n'),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    qwen_chat=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        DEFAULT_SYSTEM="You are a helpful assistant.",
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
)

SYSTEM_TEMPLATE = ConfigDict(
)

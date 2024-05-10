# Copyright (c) OpenMMLab. All rights reserved.
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_INDEX = 0
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'

# shikra constants
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

IMAGE_PLACEHOLDER = DEFAULT_IMAGE_TOKEN
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'

PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'

MASK_PLACEHOLDER = '<mask>'
MASKS_PLACEHOLDER = '<masks>'
PHRASE_ST_PLACEHOLDER_STAGE2 = '<Phrase>'
PHRASE_ED_PLACEHOLDER_STAGE2 = '</Phrase>'
CLASS_PLACEHOLDER = '<cls>'
REGION_PLACEHOLDER = '<region>'

""" Okapi constants: 
1. "xx_TOKEN" means special token and will be added into "added_tokens.json", denote as "<xxx>";
2. "xx_PLACEHOLDER" means placeholder and will be replace with processed features, denote as "[xxx]"
"""

# system: unit also used in decode process
BOT_TOKEN = "<Task>"
EOT_TOKEN = "</Task>"
BOU_TOKEN = "<Unit>"
EOU_TOKEN = "</Unit>"

# encode
VISUAL_PROMPT_PLACEHOLDER = '[VPT]'
VISUAL_PROMPT_INDEX = -300

# decode
BOV_TOKEN = "<v>" # VRT start
EOV_TOKEN = "</v>" # VRT end
VISUAL_REPRESENTATION_TOKEN = '<VRT>'
VISUAL_REFERENCE_TOKEN = '<REF>'

SPECIAL_TOKENS = [
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN, 
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    VISUAL_REPRESENTATION_TOKEN,
    VISUAL_REFERENCE_TOKEN
]

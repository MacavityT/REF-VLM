import re
from ref_vlm.utils.constants import PHRASE_ST_PLACEHOLDER_STAGE2,PHRASE_ED_PLACEHOLDER_STAGE2


import torch
from ref_vlm.utils.constants import (
    SPECIAL_TOKENS,
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN,
    PHRASE_ST_PLACEHOLDER_STAGE2,
    PHRASE_ED_PLACEHOLDER_STAGE2,
    VISUAL_REFERENCE_TOKEN,
)

def get_cot_elements(output, output_placeholders):
    st_indices = [match.start() for match in \
                    re.finditer(re.escape(PHRASE_ST_PLACEHOLDER_STAGE2), output)]
    ed_indices = [match.start() for match in \
                    re.finditer(re.escape(PHRASE_ED_PLACEHOLDER_STAGE2), output)]
    st_indices = [idx + len(PHRASE_ST_PLACEHOLDER_STAGE2) \
                    for idx in st_indices]

    # assert len(st_indices) == len(ed_indices)
    # get start and end placeholder pairs
    pairs = []
    contents = []
    stack = []
    combined = [(index, 'start') for index in st_indices] + \
        [(index, 'end') for index in ed_indices]
    combined.sort()

    cached_index = -1
    for index, type_ in combined:
        if cached_index > 0:
            contents.append(output[cached_index:index])
            cached_index = -1

        if type_ == 'start':
            stack.append(index)
        elif type_ == 'end':
            if stack:
                st_index = stack.pop()
                pairs.append((st_index, index))
            cached_index = index
    
    # last piece of content
    if cached_index > 0: contents.append(output[cached_index:])
    # assert len(contents) == len(pairs)
    # get phrase names
    p_names = []        
    p_placeholders = [PHRASE_ST_PLACEHOLDER_STAGE2, PHRASE_ED_PLACEHOLDER_STAGE2]
    removes = p_placeholders + output_placeholders
    for pair in pairs:
        start, end = pair
        phrase = output[start:end]
        for item in removes:
            phrase = phrase.replace(item, '')
        p_names.append(phrase)

    # get units and counts
    u_counts = []
    u_names = []
    for content in contents:
        counts = [content.count(placeholder) for placeholder in output_placeholders]
        idx_nonzero = [idx for idx, num in enumerate(counts) if num != 0]
        # assert len(idx_nonzero) == 1
        idx_placeholder = idx_nonzero[0]
        u_names.append(output_placeholders[idx_placeholder]) 
        u_counts.append(counts[idx_placeholder])

    return p_names, u_names, u_counts


def get_caption_text(text):
    pattern = r"<Task>.*?</Task>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\n+", "", cleaned_text)
    cleaned_text = re.sub(r"<\/?Phrase>", "", cleaned_text)
    cleaned_text = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text.strip()
    return cleaned_text


def get_matches_from_cot(text):
    pattern = r"Name:\s*(.+?)\s*Unit:\s*<Unit>(box|mask)</Unit>\s*Num:\s*(\d+)"
    matches = re.findall(pattern, text)
    return matches

def get_matches_from_text(text):
    p_names, u_names, u_counts  = get_cot_elements(text,['<REF>'])

    matches = []
    for phrase, num in zip(p_names,u_counts):
        matches.append((phrase,num))
    return matches



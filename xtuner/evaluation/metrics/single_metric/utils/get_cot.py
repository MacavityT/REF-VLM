import re
from xtuner.dataset.map_fns.dataset_map_fns.okapi_map_fn_stage2 import get_cot_elements


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

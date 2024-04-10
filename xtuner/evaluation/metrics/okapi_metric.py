import sys
import copy
import logging
from typing import Dict, Any, Union, Sequence,List

import torch
from transformers import EvalPrediction
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def post_process_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor):
    ids = copy.deepcopy(ids)  # do not modify origin preds and targets
    ids[ids < 0] = tokenizer.pad_token_id
    return ids

def decode_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor) -> Union[List[str], str]:
    assert ids.ndim in [1, 2]
    only_one_sentence = ids.ndim == 1
    if only_one_sentence:
        ids = ids.unsqueeze(0)
    ids = post_process_generate_ids(tokenizer, ids)
    res = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if only_one_sentence:
        return res[0]
    return res

class BaseComputeMetrics:
    def __init__(self, preprocessor: Dict[str, Any]):
        self.preprocessor = preprocessor
        self.tokenizer = self.preprocessor['text']

    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, Any]:
        preds, targets = eval_preds
        logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
        preds = decode_generate_ids(self.tokenizer, preds)
        targets = decode_generate_ids(self.tokenizer, targets)
        assert len(preds) == len(targets)
        return self.calculate_metric(preds, targets)

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        correct = 0
        failed = 0
        target_failed = 0
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
                continue
            if extract_pred is None:
                failed += 1
            if extract_pred == extract_target:
                correct += 1
        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
        }

    def extract_ans(self, string: str):
        raise NotImplementedError

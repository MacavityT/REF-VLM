import sys
import copy
import logging
from typing import Dict, Any, Union, Sequence,List
import torch
from transformers import EvalPrediction
from transformers import PreTrainedTokenizer
from mmengine.evaluator import BaseMetric
from xtuner.registry import BUILDER

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




# class BaseComputeMetrics(BaseMetric):
#     """
#     Base multimodal compute metrics
#     """
#     # def __init__(self, preprocessor: Dict[str, Any]):
#         # self.preprocessor = preprocessor
#         # self.tokenizer = self.preprocessor['text']
#     def __init__(self, tokenizer, *args, **kwargs):
#         self.tokenizer = BUILDER.build(tokenizer)

#     def process(self, data_batch:Any, data_samples:Sequence[dict]) -> None:
#         """Process one batch of data samples and predictions. The processed
#         results should be stored in ``self.results``, which will be used to
#         compute the metrics when all batches have been processed.

#         Args:
#             data_batch (Any): A batch of data from the dataloader.
#             data_samples (Sequence[dict]): A batch of outputs from
#                 the model.
#         """
#         raise NotImplementedError

#     def compute_metrics(self, results: list) -> dict:
#         correct = 0
#         failed = 0
#         target_failed = 0
#         for pred, target in zip(preds, targets):
#             extract_pred = self.extract_ans(pred)
#             extract_target = self.extract_ans(target)
#             if extract_target is None:
#                 target_failed += 1
#                 logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
#                 continue
#             if extract_pred is None:
#                 failed += 1
#             if extract_pred == extract_target:
#                 correct += 1
#         return {
#             'accuracy': 1.0 * correct / len(targets),
#             'target_failed': target_failed,
#             'failed': failed,
#         }

#     def extract_ans(self, string: str):
#         raise NotImplementedError
    
    # def __call__(self, eval_preds: EvalPrediction) -> Dict[str, Any]:
    #     preds, targets = eval_preds
    #     logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
    #     preds = decode_generate_ids(self.tokenizer, preds)
    #     targets = decode_generate_ids(self.tokenizer, targets)
    #     assert len(preds) == len(targets)
    #     return self.calculate_metric(preds, targets)


# TODO: 有需要修改的地方，BaseComputeMetrics的实现是不是要修改成与mmengine.evaluator中的BaseMetric相似？
"""
重写函数：
    1. process (类似于extract_anns的函数)
    2. compute_metrics:   Process one batch of data samples and predictions. The processed
                            results should be stored in `self.results`, which will be used to
                            compute the metrics when all batches have been processed.
"""
class BaseComputeMetrics(BaseMetric):
    """
    Base multimodal compute metrics
    """

    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = BUILDER.build(tokenizer)

    @staticmethod
    def find_first_zero_index(tensor):
        indices = torch.nonzero(tensor == 0)
        if indices.numel() > 0:
            return indices[0].item()
        else:
            return None

    def process(self, data_batch:Any, data_samples:Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        raise NotImplementedError

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        raise NotImplementedError
    
    def extract_ans(self, string: str):
        raise NotImplementedError

    def post_process_generate_ids(self, ids: torch.Tensor):
        ids = copy.deepcopy(ids)  # do not modify origin preds and targets
        ids[ids < 0] = self.tokenizer.pad_token_id
        return ids

    def decode_generate_ids(self, ids: torch.Tensor) -> Union[List[str], str]:
        assert ids.ndim in [1, 2]
        only_one_sentence = ids.ndim == 1
        if only_one_sentence:
            ids = ids.unsqueeze(0)
        ids = post_process_generate_ids(self.tokenizer, ids)
        res = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if only_one_sentence:
            return res[0]
        return res

    

    
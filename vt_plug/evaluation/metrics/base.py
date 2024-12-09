import sys
import os
import copy
import logging
from typing import Dict, Any, Union, Sequence,List
import torch
import jsonlines
from transformers import EvalPrediction
from transformers import PreTrainedTokenizer
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from xtuner.registry import BUILDER
from rich.console import Console
from rich.table import Table
from vt_plug.utils.constants import (
    BOT_TOKEN, EOT_TOKEN,
    BOU_TOKEN, EOU_TOKEN,
    BOV_TOKEN, EOV_TOKEN,
    SPECIAL_TOKENS
)


"""
# NOTE:重写方法，定义测评metric
Rewrite:
    1. process (类似于extract_anns的函数)
    2. compute_metrics:   Process one batch of data samples and predictions. The processed
                            results should be stored in `self.results`, which will be used to
                            compute the metrics when all batches have been processed.
"""
class BaseComputeMetrics(BaseMetric):
    """
    Base multimodal compute metrics
    """

    def __init__(self, tokenizer,stage=1,preprocessor=None, save_dir=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        self.preprocessor = preprocessor
        self.stage = stage
        self.save_dir = save_dir

    def save_outputs(self,pred,target,task_name):
        output_dict = {
            'pred': pred,
            'target': target,
        }
        with jsonlines.open(os.path.join(self.save_dir,f"output_{task_name}.jsonl"),"a") as f:
            f.write(output_dict)
            f.close()

    def post_process_generate_ids(self, ids: torch.Tensor):
        ids = copy.deepcopy(ids)  # do not modify origin preds and targets
        if ids[ids < 0].cpu().numpy().tolist() != []:
            ids[ids < 0] = self.tokenizer.pad_token_id
        return ids
    
    def decode_generate_ids(self, ids: torch.Tensor,skip_special_tokens=True) -> Union[List[str], str]:
        assert ids.ndim in [1, 2]
        only_one_sentence = ids.ndim == 1
        if only_one_sentence:
            ids = ids.unsqueeze(0)
        ids = self.post_process_generate_ids(ids)
        res = self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)
        if only_one_sentence:
            return res[0]
        return res
    
    def _print_results(self, table_metrics: dict) -> None:
        table_title = ' Caption Evaluation Results '
        table = Table(title=table_title)
        console = Console()
        table.add_column('Metrics', justify='left')
        table.add_column('Eval_results', justify='right')
        for cat, acc in table_metrics.items():
            table.add_row(cat, acc)
        with console.capture() as capture:
            console.print(table, end='')
        print_log('\n' + capture.get(), 'current')


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




    

    
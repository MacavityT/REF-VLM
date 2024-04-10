import json
import sys
from pycocoevalcap.eval import Cider, Meteor, Bleu, PTBTokenizer
from mmengine.registry.root import METRICS
from xtuner.evaluation.metrics.okapi_metric import BaseComputeMetrics


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@METRICS.register_module()
class REGCapComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def calculate_metric(self, preds, targets):
        preds = [self.extract_ans(p) for p in preds]
        preds = {i: [{"caption": x}] for i, x in enumerate(preds)}

        targets = [self.extract_ans(t) for t in targets]
        targets = {i: [{"caption": x}] for i, x in enumerate(targets)}
        json.dump({"preds": preds, "targets": targets}, open("rst.json", "w"))

        tokenizer = PTBTokenizer()
        targets  = tokenizer.tokenize(targets)
        preds = tokenizer.tokenize(preds)
        json.dump({"preds": preds, "targets": targets}, open("rst.json", "w"))
        cider_score, meteor_score, bleu_score = Cider(), Meteor(),Bleu(4)
        cider_rst, _ = cider_score.compute_score(targets, preds)
        meteor_rst, _ = meteor_score.compute_score(targets, preds)
        blue_rst, _ = bleu_score.compute_score(targets,preds)

        return {
            "CIDEr": cider_rst*100,
            "Meteor": meteor_rst,
            "BLEU4": blue_rst
        }

    def extract_ans(self, string: str):
        try:
            string = string.split("ASSISTANT: ")[-1].lower().split("</s>")[0]
            return string
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None

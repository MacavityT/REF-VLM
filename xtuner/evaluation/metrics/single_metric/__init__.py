from .cap import ImgCapComputeMetrics
from .pope import PopeComputeMetrics
from .rec import RECComputeMetrics
from .res import RESComputeMetrics
from .vqa import VQAComputeMetrics
from .cot import COTComputeMetrics
from .labels import LabelsComputeMetrics

__all__ = ['ImgCapComputeMetrics','PopeComputeMetrics',
           'RECComputeMetrics','RESComputeMetrics',
           'VQAComputeMetrics','COTComputeMetrics','LabelsComputeMetrics']
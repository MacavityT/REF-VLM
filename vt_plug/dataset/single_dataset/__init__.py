from .flickr import FlickrParser, FlickrDataset, FlickrCaptionDataset,FlickrSegmentationDataset
from .rec import RECDataset#, RECComputeMetrics
from .reg import REGDataset, GCDataset
from .caption import CaptionDataset
from .instr import InstructDataset,InstructMixDataset
from .gqa import GQADataset, GQAComputeMetrics
from .clevr import ClevrDataset
from .point_qa import Point_QA_local, Point_QA_twice, V7W_POINT, PointQAComputeMetrics
from .gpt_gen import GPT4Gen
from .vcr import VCRDataset, VCRPredDataset
from .vqav2 import VQAv2Dataset
from .vqaex import VQAEXDataset
from .pope import POPEVQADataset
from .grit import GRITDataset,GRITOfflineDataset
from .grand import GranDDataset
from .osprey import OspreyShortForm, OspreyPartLevel, OspreyLVISPosNeg, OspreyConversations, OspreyDetailedDescription
from .coco_interact import COCOInteract,COCOInteractSingle,COCOInteractSingleTask
from .cityscapes import Cityscapes
from .offline import OfflineDataset
from .dataset_templates import dataset_template_path
from .pascal import PascalDataset,PascalVoc59Dataset,PascalVoc459Dataset,PascalVocDataset
from .ade20k import ADE20k
from .coco_rem import COCOREMDataset,LVISDataset,LVISTestDataset
from .llava_g import LLAVAGrounding
from .png import PNGDataset
from .kitti import KITTIDataset
from .nyu import NYUDataset
from .hrwsi import HRWSIDataset
from .coco_keypoints import COCOKeypointsDataset,COCOKeypointsRECDataset
from .okvqa import OKVQADataset
from .res import RESDataset
from .coco_gcg_test import COCOGCG
from .openpsg import OpenPSGDataset
from .nocaps import NoCapsDataset
from .product1m import Product1MDataset
from .wukong import WuKongDataset
from .ocr import OCRCNDataset
from .meme import MEMEDataset
from .vqa import VQADataset
from .cvlue import CVLUECaptionDataset,CVLUEDialogueDataset,CVLUEVQADataset,CVLUERECDataset
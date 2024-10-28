import torch
import re
import os
from typing import List
from transformers import AutoTokenizer, AutoModel, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import numpy as np
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.structures import Boxes, ImageList, Instances
import detectron2.utils.comm as comm
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)
from torch.nn import CosineSimilarity
from torchvision.ops import box_iou
from . import openseg_classes
from .instance_evaluation import InstanceSegEvaluator
from .register_ade20k_panoptic import register_all_ade20k_panoptic,register_all_ade20k_semantic,ADE20K_150_CATEGORIES
from .register_cityscapes_panoptic import register_all_cityscapes_panoptic



def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    # panoptic segmentation
    if evaluator_type in [
        "coco_panoptic_seg",
        "ade20k_panoptic_seg",
        "cityscapes_panoptic_seg",
    ]:
        # if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # Cityscapes
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "cityscapes_panoptic_seg":
        # if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        # if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    # ADE20K
    if evaluator_type == "ade20k_panoptic_seg":
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # [1,queries, class_num]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    return semseg

def panoptic_inference(mask_cls, mask_pred, test_metadata,transform_eval=False,T=0.06):
    # scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    scores, labels = mask_cls.sigmoid().max(-1)   # [queries, class_num+1] -> [queries]  [1,3,5,4,6,,150150,150,...]
    mask_pred = mask_pred.sigmoid()
    num_classes = len(test_metadata.stuff_classes)
    keep = labels.ne(num_classes) & (scores > 0)
    if transform_eval:
        scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
    cur_scores = scores[keep]    # [mask_num]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]   # [mask_num,H,W]
    cur_mask_cls = mask_cls[keep]  # [mask_num,class_num+1]
    cur_mask_cls = cur_mask_cls[:, :-1]  # [mask_num,class_num]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks  # [mask_num,H,W]  

    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros(
        (h, w), dtype=torch.int32, device=cur_masks.device)
    segments_info = []

    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        return panoptic_seg, segments_info
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in test_metadata.thing_dataset_id_to_contiguous_id.values()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < 0.8:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(
                            pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(
                            pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

def instance_inference(mask_cls, mask_pred, test_metadata, num_queries):
    # mask_pred is already processed to have the same shape as original input
    image_size = mask_pred.shape[-2:]

    # [Q, K]
    # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    scores = mask_cls.sigmoid() 
    # if this is panoptic segmentation
    # if self.panoptic_on:
    num_classes = len(test_metadata.stuff_classes)
    # else:
    #     num_classes = len(self.test_metadata.thing_classes)
    labels = torch.arange(num_classes, device='cuda').unsqueeze(
        0).repeat(num_queries, 1).flatten(0, 1)
    scores_per_image, topk_indices = scores.flatten(
        0, 1).topk(100, sorted=False)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // num_classes
    mask_pred = mask_pred[topk_indices]

    # if this is panoptic segmentation, we only keep the "thing" classes
    # if self.panoptic_on:
    keep = torch.zeros_like(scores_per_image).bool()
    for i, lab in enumerate(labels_per_image):
        keep[i] = lab in test_metadata.thing_dataset_id_to_contiguous_id.values()

    scores_per_image = scores_per_image[keep]
    labels_per_image = labels_per_image[keep]
    mask_pred = mask_pred[keep]

    result = Instances(image_size)
    # mask (before sigmoid)
    result.pred_masks = (mask_pred > 0).float()
    result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    # Uncomment the following to get boxes from masks (this is slow)
    # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    # calculate average mask prob
    mask_scores_per_image = (mask_pred.sigmoid().flatten(
        1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image

    # result.scores are not utilized in the final evaluation metrics
    return result



class SEGDETProcessor:
    def __init__(self,task, 
                 iou_threshold=0.5, 
                 text_sim_threshold=0.5,
                 dataset_root='/data/Aaronzhu/DatasetStage2and3/ADE20k',
                 text_model_type='clip', 
                 model_path=None):
        self.task = task
        # Load pre-trained model tokenizer and model for evaluation
        self.text_model_type = text_model_type
        if text_model_type == 'bert':
            if model_path is not None:
                self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.text_model = AutoModel.from_pretrained(model_path)
            else:
                self.text_tokenizer = AutoTokenizer.from_pretrained("checkpoints/bert_base/bert-base-uncased")
                self.text_model = AutoModel.from_pretrained("checkpoints/bert_base/bert-base-uncased")
        elif text_model_type == 'clip':
            if model_path is not None:
                self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.text_model = CLIPModel.from_pretrained(model_path)
            else:
                self.text_tokenizer = AutoTokenizer.from_pretrained("/model/Aaronzhu/clip-14-336")
                self.text_model = CLIPModel.from_pretrained("/model/Aaronzhu/clip-14-336")

        self.dataset_root = dataset_root
        self.iou_threshold = iou_threshold
        self.text_sim_threshold = text_sim_threshold
        self.cos = CosineSimilarity(dim=1, eps=1e-6)
        self.test_class_features = None
        self.test_class_ids = None
        self.test_class_names = None
        if self.task in ['ade_panoptic','ade_semantic','cityscapes_panoptic']:
            self.process_config()
        elif "coco" in self.task:
            self.test_class_ids = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90
            ]
            self.test_class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

        elif "lvis" in self.task:
            with open('/code/vt_plug/xtuner/evaluation/metrics/single_metric/utils/lvis_classes.json') as f:
                self.test_class_names = json.load(f)
                f.close()
            with open('/code/vt_plug/xtuner/evaluation/metrics/single_metric/utils/lvis_class_clip.pkl','rb') as f2:
                self.test_class_features = pickle.load(f2)
                f2.close()
            self.test_class_ids = [i for i in range(len(self.test_class_names))]
            

        elif "ade20k_instance" in self.task:
            with open('/code/vt_plug/xtuner/evaluation/metrics/single_metric/utils/ade20k_instance_classes.json') as f:
                ade20k_category_dict = json.load(f)
                f.close()
            self.test_class_names = []
            self.test_class_ids = []
            for category in ade20k_category_dict:
                self.test_class_ids.append(category['id'])
                self.test_class_names.append(category['name'])

            with open('/code/vt_plug/xtuner/evaluation/metrics/single_metric/utils/ade20k_class_clip.pkl','rb') as f2:
                self.test_class_features = pickle.load(f2)
                f2.close()
        

        if self.test_class_names is not None and self.test_class_ids is not None:
            self.id_cls_map = {self.test_class_names[i]:id for i, id in enumerate(self.test_class_ids)}

    def convert_cls_txt_to_id(self,label_txt):
        label_dict = {}
        with open(label_txt,'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split(':')
                for class_name in line[1].split(','):
                    label_dict[class_name] = int(line[0])
            f.close()
        return label_dict
    
    def convert_cls_to_id(self,class_name):
        return self.id_cls_map[class_name]

    def process_config(self):
        self.cfg = get_cfg()

        self.cfg.DATASETS.PROPOSAL_FILES_TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 4 

        if self.task == 'ade_panoptic':
            register_all_ade20k_panoptic(self.dataset_root)
            self.cfg.DATASETS.TRAIN = ('openvocab_ade20k_panoptic_train')
            self.cfg.DATASETS.TEST = ('openvocab_ade20k_panoptic_val')
            self.eval_dataset_name = 'openvocab_ade20k_panoptic_val'
            self.len_data = 150

        elif self.task == 'ade_semantic':

            register_all_ade20k_semantic(self.dataset_root)

        elif self.task == 'cityscapes_panoptic':

            register_all_cityscapes_panoptic(self.dataset_root)
            self.cfg.DATASETS.TRAIN = ('openvocab_cityscapes_fine_panoptic_train')
            self.cfg.DATASETS.TEST = ('openvocab_cityscapes_fine_panoptic_val')
            self.eval_dataset_name = 'openvocab_cityscapes_fine_panoptic_val'
            self.len_data = 19
        else:
            raise NotImplementedError
        
        self.task_evaluator = build_evaluator(self.cfg,self.eval_dataset_name)

        train_metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN)
        test_metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST)

        self.test_metadata = test_metadata
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(
            train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(
            test_metadata, train_metadata)



    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                # there can be multiple synonyms for single class
                x_ = x_.split(',')
                res.append(x_)
            return res
        # get text classifier
        try:
            # it includes both thing and stuff
            class_names = split_labels(metadata.stuff_classes)
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}  # 解析嵌套列表
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(
                set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                res.append(x)
            return res, len(res)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            # how many templates for current classes
            num_templates.append(templated_classes_num)
        class_names = templated_class_names
        #print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names

    def get_bert_embedding(self,text):
        inputs = self.text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.text_model(**inputs)
        # Use the mean of the last hidden states as sentence embedding
        sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()

        return sentence_embedding

    def text_similarity_bert(self,str1,str2):
        emb1 = self.get_bert_embedding(str1)
        emb2 = self.get_bert_embedding(str2)

        return cosine_similarity([emb1], [emb2])[0, 0]
    
    def get_clip_embedding(self,text):
        if not isinstance(text,List):
            text = [text]
        inputs = self.text_tokenizer(text,padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = self.text_model.get_text_features(**inputs)
        return text_features

    def text_similarity_clip(self,str1,str2):
        if isinstance(str1,str) and isinstance(str2,str):
            inputs = self.text_tokenizer([str1, str2],padding=True, return_tensors="pt")
            with torch.no_grad():
                text_features = self.text_model.get_text_features(**inputs)
        elif isinstance(str1,torch.Tensor) and isinstance(str2,torch.Tensor):
            text_features = torch.cat([str1,str2])
        
        return self.cos(text_features[0].reshape(1,-1),text_features[1].reshape(1,-1)).cpu().item()
    
    def text_similarity(self,str1,str2):
        if self.text_model_type == 'bert':
            return self.text_similarity_bert(str1,str2)
        elif self.text_model_type == 'clip':
            return self.text_similarity_clip(str1,str2)


    def compute_mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
    def compute_iou_matrix(self, pred_masks, gt_masks):
        iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou_matrix[i, j] = self.compute_mask_iou(pred_mask, gt_mask)
        return iou_matrix
    
    def compute_miou(self, dt_labels, gt_labels, pred_boxes_masks, gt_boxes_masks, type):
        # Computing mIoU between predicted masks and ground truth masks
        iou_matrix = np.zeros((len(pred_boxes_masks), len(gt_boxes_masks)))
        if not isinstance(pred_boxes_masks,np.ndarray):
            pred_boxes_masks = np.array(pred_boxes_masks)
        if not isinstance(gt_boxes_masks,np.ndarray):
            gt_boxes_masks = np.array(gt_boxes_masks)
        if pred_boxes_masks.shape[1] == 4:
            iou_matrix = box_iou(torch.tensor(pred_boxes_masks)*1000, torch.tensor(gt_boxes_masks)*1000).cpu().numpy()
        else:
            for i, pred_box_mask in enumerate(pred_boxes_masks):
                for j, gt_box_mask in enumerate(gt_boxes_masks):
                    iou_matrix[i, j] = self.compute_mask_iou(pred_box_mask, gt_box_mask)

        paired_iou = []
        if type == 'whole':
            # One-to-one pairing and mean IoU calculation
            while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
                max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                paired_iou.append(iou_matrix[max_iou_idx])
                iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
                iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)
            
            return np.mean(paired_iou) if paired_iou else 0.0
        
        elif type == 'class':
            text_sims = np.zeros((len(dt_labels), len(gt_labels)))
            for i, dt_label in enumerate(dt_labels):
                for j, gt_label in enumerate(gt_labels):
                    if isinstance(gt_label,str):
                        text_sims[i, j] = self.text_similarity(dt_label,gt_label)
                    elif isinstance(gt_label,list):
                        max_similarity = 0
                        for single_label in gt_label:
                            similarity = self.text_similarity(dt_label,single_label)
                            if similarity > max_similarity:
                                max_similarity = similarity
                        text_sims[i, j] = max_similarity
                    else:
                        raise NotImplementedError
            assert text_sims.shape == iou_matrix.shape
            scores = []
            ids = []

            for i,row in enumerate(text_sims):
                max_value = row.max()
                indices = (row == max_value).nonzero()[0].tolist()
                scores.append(max_value)
                if len(indices) > 1:
                    max_iou = 0
                    max_ind = None
                    for ind in indices:
                        iou_ind = iou_matrix[i,ind]
                        if iou_ind >= max_iou:
                            max_iou = iou_ind
                            max_ind = ind
                    ids.append(max_ind)
                else:
                    ids.append(indices[0])

            pred_class_names = [gt_labels[index] for index in ids]
            iou_idx_list = [(i,j) for i,j in zip(np.arange(text_sims.shape[0]),ids)]

            for iou_idx in iou_idx_list:
                paired_iou.append(iou_matrix[iou_idx])

            return {
                'paired_iou': np.mean(paired_iou) if paired_iou else 0.0,
                'scores': scores,
                'pred_class_names': pred_class_names,
            }
        


    def evaluate_box_mask_miou(self,preds,targets,type,mask):
        # Load predictions

        mious = []
        for preds_per_image, targets_per_image in zip(preds,targets):


            if mask:
                pred_box_masks_per_image = preds_per_image['pred_masks']
                gt_box_masks_per_image = targets_per_image['gt_mask']
            else:
                pred_box_masks_per_image = preds_per_image['pred_boxes']
                gt_box_masks_per_image = targets_per_image['gt_boxes']                

            gt_labels = targets_per_image['gt_labels']
            pred_labels = preds_per_image['dt_labels']

            output = self.compute_miou(pred_labels, gt_labels, 
                                        pred_box_masks_per_image, gt_box_masks_per_image,type)

            if type == 'whole':
                mious.append(output)
                
            elif type == 'class':
                mious.append(output['paired_iou'])

        # Report mean IoU across all images
        mean_miou = np.mean(mious) if mious else 0.0  # If list is empty, return 0.0

        # print(f"Mean IoU (mIoU) across all images: {mean_miou:.3f}")
        return mean_miou
    
    
    def find_best_matches(self, dt_labels, gt_labels, pred_boxes_masks, gt_boxes_masks):

        '''
        gt_labels = [[cls_name1,cls_name2],cls_names]
        '''
        # find best matches per image.
        if not isinstance(pred_boxes_masks,np.ndarray):
            pred_boxes_masks = np.array(pred_boxes_masks)
        if not isinstance(gt_boxes_masks,np.ndarray):
            gt_boxes_masks = np.array(gt_boxes_masks)
        best_matches = []
        # Compute pair - wise IoU
        if pred_boxes_masks.shape[1] == 4:
            ious = box_iou(torch.tensor(pred_boxes_masks)*1000, torch.tensor(gt_boxes_masks)*1000).cpu().numpy()
        else:
            ious = self.compute_iou_matrix(pred_boxes_masks,gt_boxes_masks)

        text_sims = np.zeros((len(dt_labels), len(gt_labels)))

        for i, dt_label in enumerate(dt_labels):
            for j, gt_label in enumerate(gt_labels):
                if isinstance(gt_label,str):
                    text_sims[i, j] = self.text_similarity(dt_label,gt_label)
                elif isinstance(gt_label,list):
                    max_similarity = 0
                    for single_label in gt_label:
                        similarity = self.text_similarity(dt_label,single_label)
                        if similarity > max_similarity:
                            max_similarity = similarity
                    text_sims[i, j] = max_similarity
                else:
                    raise NotImplementedError

        # Find one-to-one matches satisfying both IoU and text similarity thresholds
        while ious.size > 0:
            max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
            if ious[max_iou_idx] < self.iou_threshold or text_sims[max_iou_idx] < self.text_sim_threshold:
                break  # No admissible pair found

            best_matches.append(max_iou_idx)

            # Remove selected annotations from consideration
            ious[max_iou_idx[0], :] = 0
            ious[:, max_iou_idx[1]] = 0
            text_sims[max_iou_idx[0], :] = 0
            text_sims[:, max_iou_idx[1]] = 0

        return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]
    
    def find_global_matches(self,dt_labels, gt_labels, pred_boxes_masks, gt_boxes_masks,topk=5):
        # find best matches compared to global classes

        if not isinstance(pred_boxes_masks,np.ndarray):
            pred_boxes_masks = np.array(pred_boxes_masks)

        best_matches = []
        # Compute pair - wise IoU
        if pred_boxes_masks.shape[1] == 4:
            ious = box_iou(torch.tensor(pred_boxes_masks)*1000, torch.tensor(gt_boxes_masks)*1000).cpu().numpy()
        else:
            ious = self.compute_iou_matrix(pred_boxes_masks,gt_boxes_masks)

        text_sims = np.zeros((len(dt_labels), len(self.test_class_names)))

        for i, dt_label in enumerate(dt_labels):
            for j, gt_cls in enumerate(self.test_class_names):
                text_sims[i, j] = self.text_similarity(dt_label,gt_cls)

        all_logits = F.softmax(torch.tensor(text_sims),dim=-1)
        cls_logits, cls_index = all_logits.topk(topk)
        cls_name_lists = self.test_class_names[cls_index]  # [mask_num, topk]

        
        for i, pred_cls_name in enumerate(cls_name_lists):
            max_iou_per_class = 0
            max_iou_idx = None
            for j, gt_cls_name in enumerate(gt_labels):
                if isinstance(gt_cls_name,str):
                    if gt_cls_name in pred_cls_name:
                        iou = ious[i,j]
                        if iou >= max_iou_per_class:
                            max_iou_per_class = iou
                            max_iou_idx = (i,j)
                elif isinstance(gt_cls_name,list):
                    for single_gt_cls_name in gt_cls_name:
                        if single_gt_cls_name in pred_cls_name:
                            iou = ious[i,j]
                            if iou >= max_iou_per_class:
                                max_iou_per_class = iou
                                max_iou_idx = (i,j)
                            break
            
            if max_iou_idx is not None:
                best_matches.append(max_iou_idx)
        
        return best_matches

    
    def evaluate_recall_with_mapping(self, preds, targets, mask, global_softmax=False):

        true_positives = 0
        actual_positives = 0

        for preds_per_image, targets_per_image in zip(preds,targets):
            try:
                if mask:
                    pred_box_masks_per_image = preds_per_image['pred_masks']
                    gt_box_masks_per_image = targets_per_image['gt_mask']
                else:
                    pred_box_masks_per_image = preds_per_image['pred_boxes']
                    gt_box_masks_per_image = targets_per_image['gt_boxes']       

                gt_labels = targets_per_image['gt_labels']
                pred_labels = preds_per_image['dt_labels']

                actual_positives += len(gt_labels)

                # Find best matching pairs
                if global_softmax:
                    best_matches = self.find_global_matches(pred_labels,gt_labels,pred_box_masks_per_image,
                                                            gt_box_masks_per_image)
                else:
                    best_matches = self.find_best_matches(pred_labels,gt_labels,pred_box_masks_per_image,
                                                            gt_box_masks_per_image)

                true_positives += len(best_matches)
            except Exception as e:
                print(e)

        recall = true_positives / actual_positives if actual_positives > 0 else 0

        return recall



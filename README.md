# REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding

<p align="center" width="100%">
<img src="images/A_00_First.jpg"  width="80%" height="80%">
</p>


 
 -----------------



Official PyTorch implementation of "REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding" [ICCV 2025 under review].

## Updates
<!-- - **28 Feb, 2024** :boom::boom: Our paper has been accepted by CVPR 2024! 🎉
- **05 Sep, 2023**: We release the code, data, and [LCL-2WAY-WEIGHT](https://huggingface.co/ISEKAI-Portal/LCL_2WAY_WEIGHT) checkpoint.
- **24 Aug, 2023**: We release the online demo at [🔗LCL-Demo🔗](http://117.144.81.99:20488/).
- **17 Aug, 2023**: We release the two subsets of ISEKAI (ISEKAI-10 and ISEKAI-pair) at [[Hugging Face 🤗]](https://huggingface.co/ISEKAI-Portal). -->

---
This repository contains the **official implementation** and **dataset** of the following paper:

> **REF-VLM: Triplet-Based Referring Paradigm for Unified Visual Decoding**<br>
>
> **Abstract:** *Multimodal Large Language Models (MLLMs) demonstrate robust zero-shot capabilities across diverse vision-language tasks after training on mega-scale datasets. However, dense prediction tasks, such as semantic segmentation and keypoint detection, pose significant challenges for MLLMs when represented solely as text outputs. Simultaneously, current MLLMs utilizing latent embeddings for visual task decoding generally demonstrate limited adaptability to both multi-task learning and multi-granularity scenarios. In this work, we present REF-VLM, an end-to-end framework for unified training of various visual decoding tasks. To address complex visual decoding scenarios, we introduce the Triplet-Based Referring Paradigm (TRP), which explicitly decouples three critical dimensions in visual decoding tasks through a triplet structure: concepts, decoding types, and targets. TRP employs symbolic delimiters to enforce structured representation learning, enhancing the parsability and interpretability of model outputs. Additionally, we construct Visual-Task Instruction Following Dataset (VTInstruct), a large-scale multi-task dataset containing over 100 million multimodal dialogue samples across 25 task types. Beyond text inputs and outputs, VT-Instruct incorporates various visual prompts such as point, box, scribble, and mask, and generates outputs composed of text and visual units like box, keypoint, depth and mask. The combination of different visual prompts and visual units generates a wide variety of task types, expanding the applicability of REF-VLM significantly. Both qualitative and quantitative experiments demonstrate that our REF-VLM outperforms other MLLMs across a variety of standard benchmarks.*

  
## Todo

1. [x] Release the training and inference code.
2. [ ] Release the checkpoints.
3. [ ] Release the VT-Instruct dataset.
4. [x] Release the demo.

## Get Start

- [Install](#install)
- [Checkpoint](#checkpoint)
- [Dataset](#dataset)
- [Demo](#demo)

## Install
### Dependencies
1. This project is built on [Xtuner](https://github.com/InternLM/xtuner). Please refer to the official documents of these toolkits for installation guidance.
2. Dataset load is base on [detectron2](https://github.com/facebookresearch/detectron2).
3. [MMDetection]()
4. [COCO 2018 Panoptic Segmentation Task API](https://github.com/cocodataset/panopticapi)

### configure accelerate

```shell
accelerate config
```
## Dataset
Coming soon.

```text
REV-VLM/
├── checkpoints
    ├── vicuna_7b
        ├──stage1
            ├──instances.json
            ├──refs(unc).p
        ├── stage2
        ├── hf_model
```

## Checkpoint
Coming soon.

```text
REV-VLM/
├── checkpoints
    ├── vicuna_7b
        ├──stage1
            ├──instances.json
            ├──refs(unc).p
        ├── stage2
        ├── hf_model
```

## Demo

To launch a Gradio web demo, use the following command. Please note that the model evaluates in the torch.float16 format, which requires a GPU with at least 16GB of memory.

```shell
python demo/app.py --config /path/to/config
```

<!-- It is also possible to use it in 8-bit quantization, albeit at the expense of sacrificing some performance.

```shell
python ./mllm/demo/demo.py --model_path /path/to/lcl/ckpt --load_in_8bit
``` -->

## Train

After preparing [data](), you can train the model using the command:

### Stage1
```shell
NPROC_PER_NODE=8 xtuner train configs/train_stage1.py --deepspeed deepspeed_zero2
```

### Stage2
```shell
NPROC_PER_NODE=8 xtuner train configs/train_stage2.py --deepspeed deepspeed_zero2
```

### Stage3
```shell
NPROC_PER_NODE=8 xtuner train configs/train_stage3_keypoint.py --deepspeed deepspeed_zero2
```

## Cite

<!-- ```bibtex
@inproceedings{
        anonymous2024vtplug,
        title={{VT}-{PLUG}: Integrating Visual Task Plugins with Unified Instruction Tuning},
        author={Anonymous},
        booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=a4PBF1YInZ},
        note={under review}
}
``` -->

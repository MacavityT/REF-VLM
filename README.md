# VT-PLUG: Integrating Visual Task Plugins with Unified Instruction Tuning

<p align="center" width="100%">
<img src="images/A_00_First.jpg"  width="80%" height="80%">
</p>


 
 -----------------



Official PyTorch implementation of "[VT-PLUG: Integrating Visual Task Plugins with Unified Instruction Tuning](https://openreview.net/forum?id=a4PBF1YInZ)" [ICLR 2025 under review].

## Updates
<!-- - **28 Feb, 2024** :boom::boom: Our paper has been accepted by CVPR 2024! 🎉
- **05 Sep, 2023**: We release the code, data, and [LCL-2WAY-WEIGHT](https://huggingface.co/ISEKAI-Portal/LCL_2WAY_WEIGHT) checkpoint.
- **24 Aug, 2023**: We release the online demo at [🔗LCL-Demo🔗](http://117.144.81.99:20488/).
- **17 Aug, 2023**: We release the two subsets of ISEKAI (ISEKAI-10 and ISEKAI-pair) at [[Hugging Face 🤗]](https://huggingface.co/ISEKAI-Portal). -->

---
This repository contains the **official implementation** and **dataset** of the following paper:

> **VT-PLUG: Integrating Visual Task Plugins with Unified Instruction Tuning**<br>
> https://openreview.net/forum?id=a4PBF1YInZ
>
> **Abstract:** *Multimodal Large Language Models (MLLMs) demonstrate robust zero-shot capabilities across diverse vision-language tasks after training on mega-scale datasets. However, dense prediction tasks, such as semantic segmentation and keypoint detection, pose significant challenges for MLLMs when represented solely as text outputs. These challenges often necessitate task-specific visual decoders, leading to the underutilization of MLLMs' multi-task potential. In this work, we propose VT-PLUG, a novel framework that leverages modular visual components as scalable plugins for a variety of visual applications. During the joint training of vision-language tasks with varying prediction densities, we propose a Visual Decoding Chain-of-Thought (VD-CoT) mechanism to prevent task conflicts. VD-CoT requires the model to predict the current task's recognition entities, decoding unit type, and other specific details, while also providing learnable queries for precise decoding. Additionally, we construct VT-Instruct, a large-scale multi-task dataset containing over 100 million multimodal dialogue samples across 25 task types. Beyond text inputs and outputs, VT-Instruct incorporates various visual prompts such as point, box, scribble, and mask, and generates outputs composed of text and visual units like point, box, keypoint, and mask. The combination of different visual prompts and visual units generates a wide variety of task types, expanding the applicability of VT-PLUG significantly.*

  
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

```shell
pip install -r requirements.txt
```

### configure accelerate

```shell
accelerate config
```
## Dataset


## Checkpoint
Coming soon.


## Demo

To launch a Gradio web demo, use the following command. Please note that the model evaluates in the torch.float16 format, which requires a GPU with at least 16GB of memory.

```shell
python  --model_path /path/to/ckpt
```

<!-- It is also possible to use it in 8-bit quantization, albeit at the expense of sacrificing some performance.

```shell
python ./mllm/demo/demo.py --model_path /path/to/lcl/ckpt --load_in_8bit
``` -->

## Train

After preparing [data](), you can train the model using the command:

### Stage1
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_train_2way_weight.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```

### Stage2
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_train_mix1.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```

### Stage3

## Inference


## Cite

```bibtex
@inproceedings{
        anonymous2024vtplug,
        title={{VT}-{PLUG}: Integrating Visual Task Plugins with Unified Instruction Tuning},
        author={Anonymous},
        booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=a4PBF1YInZ},
        note={under review}
}
```

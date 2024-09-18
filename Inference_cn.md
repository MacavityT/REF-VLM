# 模型推理使用文档


* 使用hpc镜像: **ladon-0911**


* 模型存放地址
```
/model/Aaronzhu/OkapiModel/vicuna_7b/stage2    # no ref
/model/Aaronzhu/OkapiModel/vicuna_7b/stage2_ref  # ref adapter
```

* 推理代码config
```
configs_okapi/okapi_7b_inference_stage2_decoder.py
```

### Step 1:
* 从模型存放地址中找到.pth后缀的文件夹，将该文件夹的pth模型转换为hf格式，示例：
```
# code + model_config_path + pth_path + save_dir
python xtuner/tools/model_converters/okapi_pth_to_hf.py /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0914_full_512_124/okapi_7b_train_stage2_decoder.py /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0914_full_512_124/iter_32500.pth /path/to/save/
```


### Step 2:
* 修改推理代码config文件configs_okapi/okapi_7b_inference_stage2_decoder.py，修改里面的model_dir为上述step1中所保存的hf转换后格式的路径
* 确认模型版本
  * (1) 是否有ref，如果无ref，把config中的ref_adapter以及model里的ref_adapter注释掉；
  * (2) 是否用了convnext，**0908**之前的版本都是使用clip，没有使用convnext，把model里的visual_tower注释掉，如果使用了convnext，若model name为512开头，使用clip_convnext_512['visual_encoder']，如果为320开头，使用clip_convnext_320['visual_encoder']
* 修改完config后，运行下述代码
```
cd code/okapi-mllm/
python demo/app.py
```
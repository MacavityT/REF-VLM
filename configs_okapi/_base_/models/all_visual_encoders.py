from transformers import CLIPImageProcessor, CLIPVisionModel
from xtuner.model.modules.encoder import _build_convnext_processor, CLIPConvNextModel

# openai/clip-vit-large-patch14-336
clip_patch14_336_path = '/model/Aaronzhu/clip-14-336'
clip_convnext_320_path = '/model/Aaronzhu/OkapiModel/CLIP-ConvNext'

clip_patch14_336 = dict(
    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=clip_patch14_336_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=clip_patch14_336_path)
)

clip_convnext_320 = dict(
    image_processor=dict(
        type=_build_convnext_processor,
        pretrained_model_path=clip_patch14_336_path,
        size=dict(
            shortest_edge=320
        ),
        crop_size=dict(
            height=320,
            width=320
        )
    ),
    visual_encoder=dict(
        type=CLIPConvNextModel.from_pretrained,
        pretrained_model_path=clip_convnext_320_path
    )
)
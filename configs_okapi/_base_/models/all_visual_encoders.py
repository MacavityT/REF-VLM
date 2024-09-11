from transformers import CLIPImageProcessor, CLIPVisionModel
from xtuner.model.modules.encoder import CLIPConvNextModel

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
    type=CLIPConvNextModel.from_pretrained,
    pretrained_model_path=clip_convnext_320_path
)
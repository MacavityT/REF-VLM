from transformers import CLIPImageProcessor, CLIPVisionModel

clip_patch14_336_path = 'openai/clip-vit-large-patch14-336'

clip_patch14_336 = dict(
    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=clip_patch14_336_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=clip_patch14_336_path)
)

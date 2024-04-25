from transformers import AutoModel

projector_path = ''

projector = dict(
    type=AutoModel.from_pretrained,
    projector_path=projector_path,
    trust_remote_code=True
)
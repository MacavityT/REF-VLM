import json
import jsonlines
from PIL import Image
import os


if __name__ == "__main__":
    image_dir = "/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/images"
    for root, ds, fs in os.walk(image_dir):
        for f in fs:
            fullname = os.path.join(root, f)
            image = Image.open(fullname)
            image_dict = {f"{fs}/{f}":{"width":image.width,"height":image.height}}

            print(image_dict)
            break
            with jsonlines.open("/data/Aaronzhu/DatasetStage1/Shikra/shape/llava_sbu_558k_shape.jsonl","a") as f2:
                f2.write(image_dict)
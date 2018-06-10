import os
from pathlib import Path

from PIL import Image

from demo import scale


def prepare_images(in_path, out_path):
    in_path = Path(in_path)
    out_path = Path(out_path)
    low_res_path, high_res_path, simple_high_res_path, gan_high_res_path = \
        out_path / 'low_res', out_path / 'high_res', out_path / 'simple_high_res', out_path / 'gan_high_res'

    low_res_path.mkdir(parents=True, exist_ok=True)
    high_res_path.mkdir(parents=True, exist_ok=True)
    simple_high_res_path.mkdir(parents=True, exist_ok=True)
    gan_high_res_path.mkdir(parents=True, exist_ok=True)

    for filename in in_path.iterdir():
        high_res_img = Image.open(filename)
        low_res_img = high_res_img.resize((high_res_img.size[0] // 2, high_res_img.size[1] // 2), resample=Image.BICUBIC)

        high_res_img.save(high_res_path / filename.name)
        low_res_img.save(low_res_path / filename.name)
        low_res_img.resize(high_res_img.size).save(simple_high_res_path / filename.name, resample=Image.BICUBIC)

def preparation_lfw(in_path, out_path, use_cuda=False):
    in_path = Path(in_path)
    out_path = Path(out_path)

    low_res_path, high_res_path, simple_high_res_path, gan_high_res_path = \
        out_path / 'low_res_lfw', out_path / 'high_res_lfw', \
        out_path / 'simple_high_res_lfw', out_path / 'gan_high_res_lfw'


    for i, folder in enumerate(in_path.iterdir()):
        print("{}/{}".format(i + 1, len(os.listdir(in_path))))

        (low_res_path / folder.name).mkdir(parents=True, exist_ok=True)
        (high_res_path / folder.name).mkdir(parents=True, exist_ok=True)
        (simple_high_res_path / folder.name).mkdir(parents=True, exist_ok=True)
        (gan_high_res_path / folder.name).mkdir(parents=True, exist_ok=True)

        for image in folder.iterdir():
            low_res_image_path = low_res_path / folder.name / image.name
            high_res_image_path = high_res_path / folder.name / image.name
            simple_high_res_image_path = simple_high_res_path / folder.name / image.name
            gan_high_res_image_path = gan_high_res_path / folder.name / image.name

            high_res_img = Image.open(image)
            low_res_img = high_res_img.resize((high_res_img.size[0] // 2, high_res_img.size[1] // 2),
                                              resample=Image.BICUBIC)

            high_res_img.save(high_res_image_path)
            low_res_img.save(low_res_image_path)
            low_res_img.resize(high_res_img.size).save(simple_high_res_image_path, resample=Image.BICUBIC)

            scale(image, gan_high_res_image_path, use_cuda=use_cuda)

if __name__ == "__main__":
    preparation_lfw("data/faces/lfw", "data/faces/", use_cuda=True)
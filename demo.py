from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import cv2
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

from models import Generator, Discriminator

def scale(in_file, out_file, use_cuda=False):
    low_res_img = Image.open(in_file)
    unnormalize = transforms.Compose([transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])])
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    generator = Generator(16, 2) # the second parameter is responsible for upsampling
    if use_cuda:
        generator.load_state_dict(torch.load("models/generator_improved_cifar100.pth"))
    else:
        generator.load_state_dict(torch.load("models/generator_improved_cifar100.pth", map_location=lambda storage, loc: storage))

    low_res_img = Variable(normalize(low_res_img).unsqueeze(0))

    if use_cuda:
        generator.cuda()
        low_res_img = low_res_img.cuda()


    high_res_fake = generator(low_res_img)
    save_image(unnormalize(high_res_fake.squeeze()), out_file)


if __name__ == "__main__":
    scale("data/demo/out/low_res/COCO_test2014_000000000083.jpg", "data/demo/out/gan_high_res/COCO_test2014_000000000083.jpg", use_cuda=True)
    scale("data/demo/out/low_res/COCO_test2014_000000000665.jpg", "data/demo/out/gan_high_res/COCO_test2014_000000000665.jpg", use_cuda=True)

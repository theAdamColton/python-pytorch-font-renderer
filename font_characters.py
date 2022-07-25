import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.io import read_image
import os

import bpdb


def read_image_to_tensor(filepath: str, res: int=64):
    img = read_image(filepath) # Reads as 1 by x by x
    img = img.float()
    img /= 255

    assert img.shape[1] >= res # Will only downscale
    downscale_factor = int(img.shape[1] / res)
    assert downscale_factor * res == img.shape[1]

    out = F.avg_pool2d(img, downscale_factor)
    return out


"""
filepath: path to 95 font characters png images, named 0-94.png based on their ascii code.
"""
def load_font_characters(filepath: str, res: int=32):
    char_images = torch.zeros(95, res, res)
    char_files = os.listdir(filepath)

    for char_file in char_files:
        # Gets rid of .png and converts to int
        char_index = int(os.path.basename(char_file)[:-4])
        char_tensor = read_image_to_tensor(os.path.join(filepath, char_file), res=res)

        char_images[char_index] = char_tensor
    return char_images
        


if __name__ in {"__main__", "__console__"}:
    fc = load_font_characters("./font_images/", res=32)
    bpdb.set_trace()

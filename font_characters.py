import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize, crop
import os


def read_image_to_tensor(filepath: str, res: int = 64, crop_size=20):
    img = read_image(filepath)  # Reads as 1 by x by x

    # Most font images have lots of whitespace around them
    # This step crops the images
    img = crop(
        img, crop_size, crop_size, img.shape[1] - crop_size, img.shape[2] - crop_size
    )

    img = resize(img, [res, res])
    img = img.float()
    img /= 255

    return img


"""
filepath: path to 95 font characters png images, named 0-94.png based on their ascii code.
"""


def load_font_characters(filepath: str, res: int = 32, zoom=20):
    char_images = torch.zeros(95, res, res)
    char_files = os.listdir(filepath)

    for char_file in char_files:
        # Gets rid of .png and converts to int
        char_index = int(os.path.basename(char_file)[:-4])
        char_tensor = read_image_to_tensor(
            os.path.join(filepath, char_file), crop_size=zoom, res=res
        )

        char_images[char_index] = char_tensor
    return char_images

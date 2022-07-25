import torch
import os

import bpdb

import font_characters

import matplotlib.pyplot as plt

def testplot(x):
    plt.imshow(x, cmap='gray', vmin=0, vmax=1)
    plt.show()


class FontRenderer:
    """
    res: resolution of each character in the output image
    """
    def __init__(self, res: int=32, device=torch.device('cpu')):
        font_images_path = os.path.join(os.path.dirname(__file__), "font_images/")
        self.font_characters = font_characters.load_font_characters(font_images_path, res=res)
        self.font_res = res
        self.font_characters = self.font_characters.to(device)

    """
    string_tensor: A batchsize x height x width integer tensor, where each
    pixel represents the character index, from 0 to 94
    """
    def render(self, string_tensor: torch.Tensor):
        batchsize: int = string_tensor.shape[0]
        width: int = string_tensor.shape[1]
        height: int = string_tensor.shape[2]
        characters = self.font_characters[string_tensor]
        characters_batch_by_2d = characters.movedim(2,3).reshape(batchsize,
                width * self.font_res, height * self.font_res)
        return characters_batch_by_2d


if __name__ in {"__main__", "__console__"}:

    fr = FontRenderer(res=64)
    input_t = torch.arange(0,16).reshape(1,4,4)
    res = fr.render(input_t)

    testplot(res[0])
    bpdb.set_trace()


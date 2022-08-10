import torch
import os

import font_characters


class FontRenderer:
    def __init__(self, res: int = 32, device=torch.device("cpu"), zoom=20):
        """
        res: resolution of each character in the output image
        """
        font_images_path = os.path.join(os.path.dirname(__file__), "font_images/")
        self.font_characters = font_characters.load_font_characters(
            font_images_path, res=res, zoom=zoom
        )
        self.font_res = res
        self.font_characters = self.font_characters.to(device)
        self.zoom = zoom

    def render(self, string_tensor: torch.Tensor):
        """
        string_tensor: A batchsize x height x width integer tensor, where each
        'pixel' represents the character index, from 0 to 94

        Alternatively, string_tensor can be a batchsize x 95 x height x width,
        where each pixel is a one hot encoded vector.
        """
        if len(string_tensor.shape) > 3:
            assert string_tensor.shape[1] == 95
            string_tensor = string_tensor.argmax(dim=1)
        batchsize: int = string_tensor.shape[0]
        width: int = string_tensor.shape[1]
        height: int = string_tensor.shape[2]
        # Shape: batchsize x width x height x font_res x font_res
        characters = self.font_characters[string_tensor]
        characters_batch_by_2d = characters.movedim(2, 3).reshape(
            batchsize, width * self.font_res, height * self.font_res
        )
        return characters_batch_by_2d

    def __call__(self, string_tensor: torch.Tensor):
        return self.render(string_tensor)


class ContinuousFontRenderer(FontRenderer):
    """
    At each character position in the output image, characters are a linear
    combination of the input character vector. This supports rendering input
    images which have pixels that are not one-hot-encoded.

    Unlike FontRenderer, this renderer is continuous, and supports backprop as it does not use
    argmax to produce the output image.
    """

    def render(self, string_tensor: torch.Tensor):
        """
        string_tensor: A batchsize x 95 x height x width integer tensor, where each
        'pixel' represents the character index, from 0 to 94
        """
        assert string_tensor.shape[1] == 95
        assert len(string_tensor.shape) == 4

        batchsize = string_tensor.shape[0]
        width = string_tensor.shape[2]
        height = string_tensor.shape[3]
        # Shape: batchsize x width x height x font_res x font_res
        rendered_characters = torch.tensordot(
            string_tensor, self.font_characters, dims=([1], [0])
        )
        characters_batch_by_2d = rendered_characters.movedim(2, 3).reshape(
            batchsize, width * self.font_res, height * self.font_res
        )
        return characters_batch_by_2d


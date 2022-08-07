import torch
import matplotlib.pyplot as plt

from font_renderer import FontRenderer, ContinuousFontRenderer


def testplot(x):
    plt.imshow(x, cmap="gray", vmin=0, vmax=1)
    plt.show()


if __name__ in {"__main__", "__console__"}:
    fr = FontRenderer(res=64)
    input_t = torch.arange(0, 16).reshape(1, 4, 4)
    res = fr.render(input_t)

    testplot(res[0])

    frc = ContinuousFontRenderer(res=16)
    input_t = torch.zeros(1, 95, 4, 4)
    input_t[0,32,:,:] = 0.5
    input_t[0,63,:,:] = 0.5
    res = frc(input_t)

    testplot(res[0])

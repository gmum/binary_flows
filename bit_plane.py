import torch
import matplotlib.pyplot as plt


class Img2BP(object):
    '''
    Torchvision transform:
    Convert Image (tensor) to Bit plane (tensor)
    IMPORTANT: Transform assumes no batch dimension!
    '''

    def __init__(self, bits=8):
        assert isinstance(bits, int)
        self.bits = bits

    def __call__(self, x):
        y = torch.fmod(x, 2)
        for b in range(1, self.bits):
            y = torch.cat((y, torch.fmod(torch.floor(x / 2 ** b), 2)), 0)
        return y


def bp2img(x, bits=8):
    for i in range(bits):
        if i == 0:
            x_img = x[:, [0]]
        else:
            x_img += (2 ** i) * x[:, [i]]
    return x_img


def img2bp(x, bits=8):
    # image to bit plane
    y = torch.fmod(x, 2)
    for b in range(1, bits):
        y = torch.cat((y, torch.fmod(torch.floor(x/2**b), 2)), 1)
    return y


def plot_bit_plane(x, bits=8):
    # from image to bit plane
    y = img2bp(x, bits=bits)
    # reconstruction using only 3 most significant bits
    z = 2 * (2 * (2 * y[:,-1] + y[:,-2]) + y[:,-3])
    # to plot
    to_plot = [x.squeeze()]
    for i in range(bits):
        to_plot.append(y[:,i].squeeze())
    to_plot.append(z.squeeze())
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, i in zip(axes.flat, to_plot):
        ax.imshow(i, cmap='gray')
    plt.tight_layout()
    plt.show()

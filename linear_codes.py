import torch

# The generator matrix for Hamming(7,4,3)
G_hamming743 = torch.tensor([[1., 1., 0., 1.],
                             [1., 0., 1., 1.],
                             [1., 0., 0., 0.],
                             [0., 1., 1., 1.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
G_hamming743 = G_hamming743.unsqueeze(0)

# The generator matrix for Hadamard(8,3,4)
G_hadamard834 = torch.tensor([[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 1., 0.],
                              [0., 0., 1.],
                              [1., 1., 0.],
                              [1., 0., 1.],
                              [0., 1., 1.],
                              [1., 1., 1.]])
G_hadamard834 = G_hadamard834.unsqueeze(0)


def hamming743(x):
    # 4 most significant bits
    x_ms_bits = x[:, 4:]
    shape_x = x_ms_bits.shape

    # linear code
    y = torch.fmod(torch.matmul(G_hamming743, x_ms_bits.flatten(2)).view(shape_x[0], 7, *shape_x[2:]), 2)
    return y


def hadamard834(x):
    # 3 most significant bits
    x_ms_bits = x[:, 5:]
    shape_x = x_ms_bits.shape

    # linear code
    y = torch.fmod(torch.matmul(G_hadamard834, x_ms_bits.flatten(2)).view(shape_x[0], 8, *shape_x[2:]), 2)
    return y


class HadamardCode(object):
    '''
    Torchvision transform:
    Add Hadamard(8,3,4) to the input.
    IMPORTANT: Transform assumes no batch dimension!
    '''

    def __init__(self, bits=8):
        self.G_hadamard = torch.tensor([[0., 0., 0.],
                                        [1., 0., 0.],
                                        [0., 1., 0.],
                                        [0., 0., 1.],
                                        [1., 1., 0.],
                                        [1., 0., 1.],
                                        [0., 1., 1.],
                                        [1., 1., 1.]])

        self.bits = bits

    def __call__(self, x):
        # take three most significant bits
        x_ms_bits = x[self.bits//2 + 1:]
        shape_x = x_ms_bits.shape

        # linear code
        y = torch.fmod(torch.matmul(self.G_hadamard, x_ms_bits.flatten(1)).view(self.bits, *shape_x[1:]), 2)

        # concatenate x (B x 8 x W x H) and y (B x 8 x W x H) -> B x 16 x W x H)
        z = torch.cat((y, x), 0)
        return z


if __name__ == '__main__':

    f = HadamardCode()
    x = torch.randn(1, 8, 8, 8)
    print(f(x).shape)
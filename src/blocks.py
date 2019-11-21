from torch.nn import (
    ConvTranspose2d, Conv2d,
    BatchNorm2d, ReLU, LeakyReLU
)


def generator_conv_block(n_input, n_output, k_size = 4, stride = 2, padding = 0):
    '''Generator Convolutional Block
    Reference: https://arxiv.org/pdf/1807.00734.pdf
    Params:
        n_input     -> Number of Input Channels
        n_output    -> Number of Output Channels
        k_size      -> Size of the convolution kernel
        stride      -> Stride of the convolution kernel
        padding     -> Amount of padding
    '''
    block = [
        ConvTranspose2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            bias=False
        ),
        BatchNorm2d(n_output),
        ReLU(inplace=True),
    ]
    return block



def discriminator_conv_block(n_input, n_output, k_size = 4, stride = 2, padding = 0, bn = False):
    '''Discriminator Convolutional Block
    Reference: https://arxiv.org/pdf/1807.00734.pdf
    Params:
        n_input     -> Number of Input Channels
        n_output    -> Number of Output Channels
        k_size      -> Size of the convolution kernel
        stride      -> Stride of the convolution kernel
        padding     -> Amount of padding
        bn          -> Apply BatchNormalization or not
    '''
    block = [Conv2d(
        n_input, n_output,
        kernel_size=k_size,
        stride=stride,
        padding=padding,
        bias=False
    )]
    if bn:
        block.append(BatchNorm2d(n_output))
    block.append(LeakyReLU(0.2, inplace=True))
    return block
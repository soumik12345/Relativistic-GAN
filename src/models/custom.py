from ..blocks import *
from torch.nn import (
    Module, ConvTranspose2d, Conv2d,
    BatchNorm2d, ReLU, Tanh, Sequential
)


class Generator(Module):

    def __init__(self, latent_dimension = 128, channels = 3):
        '''Custom Generator Architecture
        Reference: https://arxiv.org/pdf/1807.00734.pdf
        Params:
            latent_dimension -> Input Dimension to Generator
            channels         -> Number of Channels in the Image
        '''
        super(Generator, self).__init__()
        # Parameter Initializations
        self.latent_dimension = latent_dimension
        self.channels = channels
        # Generator Model
        self.model = Sequential(
            *generator_conv_block(self.latent_dimension, 1024, 4, 1, 0),
            *generator_conv_block(1024, 512, 4, 2, 1),
            *generator_conv_block(512, 256, 4, 2, 1),
            *generator_conv_block(256, 128, 4, 2, 1),
            *generator_conv_block(128, 64, 4, 2, 1),
            ConvTranspose2d(64, self.channels, 3, 1, 1),
            Tanh()
        )
    
    def forward(self, z):
        '''Forward Pass'''
        z = z.view(-1, self.latent_dimension, 1, 1)
        image = self.model(z)
        return image



class Discriminator(Module):

    def __init__(self, channels = 3):
        '''Custom Discriminator Architecture
        Reference: https://arxiv.org/pdf/1807.00734.pdf
        Params:
            channels -> Number of Channels in the Image
        '''
        super(Discriminator, self).__init__()
        # Parameter Initializations
        self.channels = channels
        # Discriminator Model
        self.model = Sequential(
            *discriminator_conv_block(self.channels, 32, 4, 2, 1),
            *discriminator_conv_block(32, 64, 4, 2, 1),
            *discriminator_conv_block(64, 128, 4, 2, 1, bn = True),
            *discriminator_conv_block(128, 256, 4, 2, 1, bn = True),
            Conv2d(256, 1, 4, 1, 0, bias = False),
        )
    
    def forward(self, images):
        '''Forward Pass'''
        out = self.model(images)
        return out.view(-1, 1)
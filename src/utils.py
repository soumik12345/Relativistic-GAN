from config import *
from torch import device, randn
from torch.cuda import is_available


def get_device():
    '''Get Device'''
    return device("cuda" if is_available() else "cpu")


def visualize_results(netG, device):
    '''Visualize Generator Results
    Params:
        netG    -> Generator Network
        device  -> Device
    '''
    noise = torch.randn(1, LATENT_DIMENSION, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    plt.imshow((gen_image + 1) / 2)
    plt.show()
from torch import device
from torch.cuda import is_available


def get_device():
    '''Get Device'''
    return device("cuda" if is_available() else "cpu")
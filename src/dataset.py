from config import *
from PIL import Image
from os import listdir
from os.path import join
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import (
    Compose, Resize,
    ToTensor, Normalize,
    CenterCrop, RandomRotation,
    RandomHorizontalFlip, RandomApply
)


class CustomDataset(Dataset):
    
    def __init__(self, img_dir, transform1=None, transform2=None):
        '''Custom Dataset
        Params:
            img_dir     -> Image Directory
            transform1  -> Transform 1
            transform2  -> Transform 2
        '''
        self.img_dir = img_dir
        self.img_names = listdir(img_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        self.imgs = []
        for img_name in self.img_names:
            img = Image.open(join(img_dir, img_name))
            if self.transform1 is not None:
                img = self.transform1(img)
            self.imgs.append(img)

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform2 is not None:
            img = self.transform2(img)
        return img

    def __len__(self):
        return len(self.imgs)


def get_transforms():
    '''Get Transforms'''
    transform1 = Compose([
        Resize(IMAGE_SIZE),
        CenterCrop(IMAGE_SIZE)
    ])
    random_transforms = [RandomRotation(degrees = 5)]
    transform2 = Compose([
        RandomHorizontalFlip(p = 0.5),
        RandomApply(random_transforms, p = 0.3),
        ToTensor(),
        Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    return transform1, transform2


def get_data_loader(transform1, transform2):
    '''Get DataLoader'''
    transform1, transform2 = get_transforms()
    train_dataset = CustomDataset(
        img_dir = IMAGE_DIRECTORY,
        transform1 = transform1,
        transform2 = transform2
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )
    return train_loader


def visualize(data_loader):
    '''Visualize Dataset'''
    imgs = next(iter(data_loader))
    imgs = imgs.numpy().transpose(0, 2, 3, 1)
    fig = plt.figure(figsize = (25, 16))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(4, 8, i + 1, xticks = [], yticks = [])
        plt.imshow((img + 1) / 2)
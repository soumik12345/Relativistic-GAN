import torch, time
from config import *
from .utils import *
from tqdm import tqdm
from torch.optim import Adam


def get_optimizers(netG, netD):
    '''Get Optimizers
    Params:
        netG -> Generator Network
        netD -> Discriminator Network
    '''
    optimizerD = Adam(
        netD.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA_1, 0.999)
    )
    optimizerG = Adam(
        netG.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA_1, 0.999)
    )
    return optimizerG, optimizerD


def train(netG, netD, optimizerG, optimizerD, data_loader, device):
    '''Training Function
    Params:
        netG        -> Generator Network
        netD        -> Discriminator Network
        optimizerG  -> Generator Optimizer
        optimizerD  -> Discriminator Optimizer
        data_loader -> Data Loader
        device      -> Device
    '''
    g_loss_history, d_loss_history = [], []
    for epoch in range(1, EPOCHS + 1):
        print('Epoch', str(epoch), 'going on....')
        start = time.time()
        for index, real_images in tqdm(enumerate(data_loader), total=len(data_loader)):
            netD.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size, 1), real_label, device=device)
            outputR = netD(real_images)
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            outputF = netD(fake.detach())
            errD = (torch.mean((outputR - torch.mean(outputF) - labels) ** 2) + 
                    torch.mean((outputF - torch.mean(outputR) + labels) ** 2))/2
            errD.backward(retain_graph=True)
            optimizerD.step()
            netG.zero_grad()
            outputF = netD(fake)   
            errG = (torch.mean((outputR - torch.mean(outputF) + labels) ** 2) +
                    torch.mean((outputF - torch.mean(outputR) - labels) ** 2))/2
            errG.backward()
            optimizerG.step()
            if (index + 1) % (len(data_loader) // 2) == 0:
                g_loss_history.append(errG.item())
                d_loss_history.append(errD.item())
        print('Discriminator Loss:{}\tGenerator Loss:{}'.format(d_loss_history[-1], g_loss_history[-1]))
        print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))
        if epoch % 10 == 0:
            visualize_results(netG, device)
    return g_loss_history, d_loss_history, netG
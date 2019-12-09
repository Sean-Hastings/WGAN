# External Code
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from matplotlib import pyplot as plt

# My Code
from models import mlp
from models import mlp0

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--img_dir', type=str, default='imgs', help='image dir to save')
    parser.add_argument('--z_size', type=int, default=32, help='noise size')
    parser.add_argument('--width', type=int, default=512, help='generator and discriminator widths')
    parser.add_argument('--depth', type=int, default=3, help='generator and discriminator depths')
    parser.add_argument('--bias', type=bool, default=True, help='Whether to use bias terms')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--disc_updates', type=int, default=10, help='number of discriminator updates per generator update')
    parser.add_argument('--alpha_g', type=float, default=0.00005, help='learning rate for generator')
    parser.add_argument('--alpha_d', type=float, default=0.00005, help='learning rate for generator')
    parser.add_argument('--info_lambda', type=float, default=0.25, help='Weighting for InfoGAN objective')

    args = parser.parse_args()

    data = datasets.FashionMNIST(root='dataset',
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0, 0, 0), (0.5, 0.5, 0.5)),
                                ]))
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    #generator = mlp0.MLP_G(28, 100, 1, 512, 1).cuda()
    #discriminator = mlp0.MLP_D(28, 100, 1, 512, 1).cuda()
    generator     = mlp.Generator(args.z_size, 28*28, args.width, args.depth, args.bias).cuda()
    #discriminator = mlp.Discriminator(28*28, args.width, args.depth, args.bias).cuda()
    discriminator = mlp.InfoDiscriminator(args.z_size, 28*28, args.width, args.depth, args.bias).cuda()

    latent_loss_fun = NormalNLLLoss()

    opt_gen  = optim.Adam(list(generator.parameters())+list(discriminator.latent_params()), args.alpha_g)
    opt_disc = optim.Adam(discriminator.parameters(), args.alpha_d)

    one = torch.FloatTensor([1]).cuda()
    zero = torch.FloatTensor([0]).cuda()

    const_z = torch.FloatTensor(5*5, args.z_size).normal_(0, 1).cuda()
    for i in range(5):
        for j in range(5):
            const_z[5*i+j, 0] = (i - 2) / 2
            const_z[5*i+j, 1] = (j - 2) / 2

    for epoch in range(args.epochs):
        print('Working on epoch %d' % (epoch+1), end='\r')
        for i, batch in enumerate(dataloader):
            images = batch[0].view(batch[0].size(0), -1).cuda()

            for p in discriminator.parameters():
                p.data.clamp_(-0.001, 0.001)

            opt_disc.zero_grad()
            d_images, _ = discriminator(images)

            z = torch.FloatTensor(images.size(0), args.z_size).normal_(0, 1).cuda()
            g_z = generator(z)
            d_gg, _ = discriminator(g_z)
            err = d_images - d_gg
            err.backward()
            opt_disc.step()

            if (i+1) % (args.disc_updates if epoch > 10 else 100) == 0:
                opt_gen.zero_grad()
                z = torch.FloatTensor(images.size(0), args.z_size).normal_(0, 1).cuda()
                g_z = generator(z)
                d_g, latent_codes = discriminator(g_z)
                latent_loss = latent_loss_fun(latent_codes[:, :2], zero, one)
                (d_g + args.info_lambda*latent_loss).backward()
                opt_gen.step()

                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, args.epochs, i, len(dataloader), (i+1)//(args.disc_updates if epoch > 10 else 100),
                err.data[0], d_g.data[0], d_images.data[0], d_gg.data[0]))

        if (epoch+1) % 25 == 0:
            with torch.no_grad():
                const_gen = generator(const_z)
            c_im = const_gen.view(const_gen.size(0), 28, 28)
            plt.figure(1)
            for i in range(5*5):
                plt.subplot(5, 5, i+1)
                plt.imshow(c_im[i].cpu().numpy(), cmap='gray')
            plt.savefig('%s/epoch_%d.png' % (args.img_dir, epoch))




































pass

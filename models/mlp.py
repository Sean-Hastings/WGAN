import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_size, out_size, width, depth, bias=True):
        nn.Module.__init__(self)
        self.in_size  = in_size
        self.out_size = out_size
        self.width    = width
        self.depth    = depth
        self.bias     = bias

        if depth > 0:
            self.network = [nn.Linear(in_size, width, bias=bias), nn.ReLU(True)]
            for i in range(depth-1):
                self.network += [nn.Linear(width, width, bias=bias), nn.ReLU(True)]
            self.network += [nn.Linear(width, out_size, bias=bias)]
            self.network = nn.Sequential(*self.network)
        else:
            self.network = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, inputs):
        return self.network(inputs)


class Generator(nn.Module):
    def __init__(self, z_size, out_size, width, depth, bias=True):
        nn.Module.__init__(self)
        self.z_size   = z_size
        self.out_size = out_size
        self.width    = width
        self.depth    = depth
        self.bias     = bias

        self.network = MLP(z_size, out_size, width, depth, bias)

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self, in_size, width, depth, bias=True):
        nn.Module.__init__(self)
        self.in_size = in_size
        self.width   = width
        self.depth   = depth
        self.bias    = bias

        self.network = MLP(in_size, 1, width, depth, bias)

    def forward(self, samples):
        return self.network(samples)


class InfoDiscriminator(nn.Module):
    def __init__(self, z_size, in_size, width, depth, bias=True):
        nn.Module.__init__(self)
        self.z_size  = z_size
        self.in_size = in_size
        self.width   = width
        self.depth   = depth
        self.bias    = bias

        self.body_network = MLP(in_size, width, width, depth-1, bias)
        self.disc_head    = nn.Sequential(nn.ReLU(), nn.Linear(width, 1, bias=bias))
        self.latent_head  = nn.Sequential(nn.ReLU(), nn.Linear(width, z_size, bias=bias))

    def parameters(self):
        return list(self.body_network.parameters()) + list(self.disc_head.parameters())

    def latent_params(self):
        return self.latent_head.parameters()

    def forward(self, samples):
        x = self.body_network(samples)
        return self.disc_head(x).mean(0).view(1), self.latent_head(x)

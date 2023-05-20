import torch
import torch.nn as nn
import torch.optim as optim

import os
from abc import abstractmethod

from base_model import BaseModel

class Vanilla_GAN(BaseModel):

    def __init__(self, args, fabric):
        super().__init__()

        self.fabric = fabric

        self.loss_names = ['net_g', 'net_g_real', 'net_g_fake', 'idt']
        self.visual_names = ['real_a', 'fake_b', 'real_b', 'idt_b']
        
        if args.phase == 'train':
            self.model_names = ['g', 'd']
        else: # during test time only load generator
            self.model_names = ['g']

        # define all networks
        self.net_g = Generator(args)
        self.init_params(self.net_g)

        if args.phase == 'train':
            self.net_d = Discriminator(args)
            self.init_params(self.net_d)

            # define loss functions # TODO: abstract this out
            self.criterionGAN = nn.BCELoss()
            self.criterionIdt = nn.L1Loss()

            # initialize optimizers
            self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
 
            # feed model and optimizers to fabric
            self.net_g, self.optimizer_g = self.fabric.setup(self.net_g, self.optimizer_g)
            self.net_d, self.optimizer_d = self.fabric.setup(self.net_d, self.optimizer_d)


    def init_params(self, m: nn.Module):
        # custom weights initialization
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def set_input(self, input): 
        """Unpack input data from dataloader and perform necessary pre-processing steps
        
        Parameters
        ----------
        input : dict
            Dictionary containing the input data
        """
        self.real_a = input['A']
        self.real_b = input['B']
        self.image_paths = input['A_paths']

    @abstractmethod
    def set_epoch(self, epoch):
        """Set the current epoch
        
        Parameters
        ----------
        epoch : int
            Current epoch
        """
        pass

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_b = self.net_g(self.real_a)
        self.idt_b = self.net_g(self.real_b)

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def data_dependent_initialize(self, data):
        """Perform data-dependent initialization"""
        self.set_input(data)
        self.forward()

    def save_networks(self, path):
        """Save all the networks to the disk."""
        state = {
            "net_g": self.net_g,
            "net_d": self.net_d,
            "optimizer_g": self.optimizer_g,
            "optimizer_d": self.optimizer_d,
            "epoch": self.current_epoch,
        }

        # save each loss function to state
        for loss_name in self.loss_names:
            state[loss_name] = getattr(self, loss_name)
        
        self.fabric.save(state, path)

    def load_networks(self):
        """Load all the networks from the disk."""

        # set epoch to whatever loaded network has
        state = {
            "net_g": self.net_g,
            "net_d": self.net_d,
            "optimizer_g": self.optimizer_g,
            "optimizer_d": self.optimizer_d,
        }
        path = os.path.join(self.args.ckpt_dir, self.args.ckpt_name)
        remainder = self.fabric.load(path, state)
        self.current_epoch = remainder['epoch']

        # assign each self loss function to remainer loss function
        for loss_name in self.loss_names:
            setattr(self, loss_name, remainder[loss_name])

        # handle edge case if losses are missing from remainder
        for loss_name in self.loss_names:
            if loss_name not in remainder:
                setattr(self, loss_name, None)

        # throw error edge case if there are too many losses in remainder
        for loss_name in remainder:
            if loss_name not in self.loss_names:
                raise ValueError("{} in remainder but not in self.loss_names".format(loss_name))

        return remainder['epoch']

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if self.args.verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (args.ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (args.ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (args.ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (args.ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (args.nc) x 64 x 64
        )

        self.main.apply(self.init_params)


    def forward(self, input: torch.Tensor):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.main = nn.Sequential(
            # input is (args.nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.main.apply(self.init_params)


    def forward(self, input: torch.Tensor):
        return self.main(input)

    



# Then, adjust vanilla_gan
# Then, adjust train
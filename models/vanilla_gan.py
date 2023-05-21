import torch
import torch.nn as nn
import torch.optim as optim

import os
from abc import abstractmethod

from models.base_model import BaseModel
from models.networks import define_G, define_D, GANLoss

def init_params(m: nn.Module):
    # custom weights initialization
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Vanilla_GAN(BaseModel):

    def __init__(self, args, fabric):
        super().__init__(args)

        self.fabric = fabric

        self.loss_names = ['loss_g', 'loss_d_real', 'loss_d_fake', 'loss_idt']
        self.visual_names = ['real_a', 'fake_b', 'real_b', 'idt_b']
        
        if args.phase == 'train':
            self.model_names = ['g', 'd']
        else: # during test time only load generator
            self.model_names = ['g']

        # define all networks
        self.net_g = define_G(args)
        self.net_g.apply(init_params)

        if args.phase == 'train':
            self.net_d = define_D(args)
            self.net_d.apply(init_params)

            # define loss functions
            self.criterionGAN = GANLoss(self.args.gan_loss, self.fabric)
            self.criterionIdt = nn.L1Loss()

            # initialize all losses to None
            for loss_name in self.loss_names:
                setattr(self, loss_name, None)

            # initialize optimizers
            self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
 
            # feed model and optimizers to fabric
            self.net_g, self.optimizer_g = self.fabric.setup(self.net_g, self.optimizer_g)
            self.net_d, self.optimizer_d = self.fabric.setup(self.net_d, self.optimizer_d)
            
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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_b = self.net_g(self.real_a)
        self.idt_b = self.net_g(self.real_b)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        
        self.forward()

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) to train discriminator
        self.set_requires_grad(self.net_d, True)
        self.optimizer_d.zero_grad() 
        self.loss_d = self.compute_loss_d()
        self.fabric.backward(self.loss_d, model=self.net_d)
        self.optimizer_d.step()

        # (2) Update G network: maximize log(D(G(z)))
        self.set_requires_grad(self.net_d, False)
        self.set_requires_grad(self.net_g, True)
        self.optimizer_g.zero_grad()
        self.loss_g = self.compute_loss_g()
        self.fabric.backward(self.loss_g, model=self.net_g)
        self.optimizer_g.step()

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

        # save each loss function to state: "loss_" + loss_name
        for loss_name in self.loss_names:
            state[loss_name] = getattr(self, loss_name)
        
        self.fabric.save(path, state)

    def load_networks(self):
        """Load all the networks from the disk."""

        # Set the epoch to whatever loaded network has
        state = {
            "net_g": self.net_g,
            "net_d": self.net_d,
            "optimizer_g": self.optimizer_g,
            "optimizer_d": self.optimizer_d,
            "epoch": 0,
        }
        
        # Load the checkpoint
        remainder = self.fabric.load(self.args.ckpt_full_path, state)
        
        # Put existing losses into state
        for loss_name in self.loss_names:
            if loss_name in remainder:
                # Add the loaded loss to the state
                state[loss_name] = remainder[loss_name]
            else:
                raise ValueError(f"Loss {loss_name} is in self.loss_names but not found in the loaded checkpoint.")
        
        self.current_epoch = state['epoch']

        # Check for extra losses in the loaded checkpoint
        for loss_name in remainder:
            if loss_name not in state:
                raise ValueError(f"{loss_name} is in the loaded checkpoint but not in self.loss_names.")

        return state['epoch']

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

    def compute_loss_d(self):
        fake = self.fake_b.detach()

        # train with fake data
        pred_fake = self.net_d(fake)
        self.loss_d_fake = self.criterionGAN(pred_fake, False).mean()

        # train with real data
        pred_real = self.net_d(self.real_b)
        self.loss_d_real = self.criterionGAN(pred_real, True).mean()

        # combine loss and calculate gradients
        return (self.loss_d_fake + self.loss_d_real) * 0.5

    # create a method that calculates the generator's loss
    def compute_loss_g(self):
        self.loss_g = self.criterionGAN(self.net_d(self.fake_b), True).mean() * self.args.lambda_g
        self.loss_idt = self.criterionIdt(self.idt_b, self.real_b) * self.args.lambda_idt
        return self.loss_g + self.loss_idt
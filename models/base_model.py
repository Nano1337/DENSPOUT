import os
import torch 
from abc import ABC, abstractmethod

class BaseModel(torch.nn.Module, ABC):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_epoch = 0
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.opt_names = []
        self.image_paths = []

    @abstractmethod 
    def set_input(self, input): 
        """Unpack input data from dataloader and perform necessary pre-processing steps
        
        Parameters
        ----------
        input : dict
            Dictionary containing the input data
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def data_dependent_initialize(self):
        """Perform data-dependent initialization"""
        pass

    @abstractmethod
    def save_networks(self):
        """Save all the networks to the disk."""
        pass

    @abstractmethod
    def load_networks(self, epoch):
        """Load all the networks from the disk."""
        pass

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

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
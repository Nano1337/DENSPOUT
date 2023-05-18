import random 
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """

    def __init__(self, args): 
        """Initialize the class; save the options in the class
        
        Parameters: 
            args (argparse.ArgumentParser): stores all the experiment flags; see utils/opt.py
        """ 
        self.args = args
        self.root = args.dataroot
        self.current_epoch = 0

        @abstractmethod
        def __len__(self):
            """Return the total number of images."""
            return 0

        @abstractmethod
        def __getitem__(self, index):
            """Return a data point and its metadata information.
            
            Parameters: 
                index (int): a random integer for data indexing

            Returns:   
                a dictionary of data with their names. It usually contains the data itself and its metadata information.
            """
            pass




        
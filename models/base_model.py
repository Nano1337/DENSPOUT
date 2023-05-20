import os
import torch 
from abc import ABC, abstractmethod

class BaseModel(torch.nn.Module, ABC):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_epoch = 0  
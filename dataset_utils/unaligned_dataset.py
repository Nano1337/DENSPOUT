import os
import random
from PIL import Image
from base_dataset import BaseDataset, get_transform
from torchvision.datasets import ImageFolder

# TODO: should be easy fix to abstract to two folders of different sizes

class UnalignedDataset(BaseDataset): 

    def __init__(self, args): 
        BaseDataset.__init__(self, args)
        self.dir_A = os.path.join(args.dataroot, args.phase + 'A')
        self.dir_B = os.path.join(args.dataroot, args.phase + 'B')
        
        if args.phase == "test" and os.path.exists(self.dir_A) \
            and os.path.exists(os.path.join(args.dataroot, "testA")):
            self.dir_A = os.path.join(args.dataroot, "testA")
            self.dir_B = os.path.join(args.dataroot, "testB") 

        self.A_imgs = ImageFolder(self.dir_A)[0]
        self.B_imgs = ImageFolder(self.dir_B)[0]

        self.transform_A = get_transform('A')
        self.transform_B = get_transform('B')

    def __len__(self):
        # We take the length of A_imgs assuming that both A and B have same length
        return len(self.A_imgs) # TODO: take max of A and B lengths for more general case

    def __getitem__(self, index):
        # Note: this assumes that there are the same number of images in both directories.
        # If the directories have different numbers of images, you might need to handle this differently.
        A_path = self.A_imgs[index]
        B_path = random.choice(self.B_imgs)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        return {'A': self.transform_A(A_img), 'B': self.transform_B(B_img), 'A_paths': A_path, 'B_paths': B_path}


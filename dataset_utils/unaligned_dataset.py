import os
import random
from PIL import Image
from dataset_utils.base_dataset import BaseDataset

class UnalignedDataset(BaseDataset): 

    def __init__(self, args, data_path): 
        BaseDataset.__init__(self, args)

        self.dir_A = os.path.join(data_path, args.phase + 'A')
        self.dir_B = os.path.join(data_path, args.phase + 'B')
        
        if args.phase == "test" and os.path.exists(self.dir_A) \
            and os.path.exists(os.path.join(data_path, "testA")):
            self.dir_A = os.path.join(data_path, "testA")
            self.dir_B = os.path.join(data_path, "testB") 

        self.A_imgs = sorted([os.path.join(self.dir_A, img) for img in os.listdir(self.dir_A) if img.endswith(('.jpg', '.png'))]) 
        self.B_imgs = sorted([os.path.join(self.dir_B, img) for img in os.listdir(self.dir_B) if img.endswith(('.jpg', '.png'))])  

        self.transform_A = self.get_transform('A')
        self.transform_B = self.get_transform('B')
        
        self.len_A = len(self.A_imgs)
        self.len_B = len(self.B_imgs)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, index):
        A_path = self.A_imgs[index % len(self.A_imgs)]
        B_path = self.B_imgs[random.randint(0, len(self.B_imgs) - 1)]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        return {'A': self.transform_A(A_img), 'B': self.transform_B(B_img), 'A_paths': A_path, 'B_paths': B_path}

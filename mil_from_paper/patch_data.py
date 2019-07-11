import os
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import random

#Though the name of the class is PatchMethod, this does not create patches out of the images!

class PatchMethod(torch.utils.data.Dataset):
    def __init__(self, root = '/home/swagatam/from_sahyadri_with_cpctr/CPCTR_4/check', mode = 'train', transform = None):
        self.root = root
        self.mode = mode
        self.raw_samples = glob.glob(root + '/*/*')
        self.samples = []
        for raw_sample in self.raw_samples:
            self.samples.append((raw_sample, int(raw_sample.split('/')[-2])))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            random.shuffle(self.samples)
            
        image_dir, label = self.samples[index]
        images = glob.glob(image_dir + '/*')
        
        transformations = transforms.Compose([
            transforms.CenterCrop(1200), #Centercropping can be avoided, but gave CUDA out of memory error in my case.
            transforms.ToTensor()
        ])
        
        array = []
        
        for i, image_path in enumerate(images):
            image = Image.open(image_path)
            image = transformations(image)
            array.append(image)
        array = tuple(array)
        array = torch.stack(array, 0)
        
        return (array, label)

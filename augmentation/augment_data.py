import os
import glob
import torch
from PIL import Image
from torchvision import transforms
import random
import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    return list(chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1)))

def _augment(root = '/home/intern_swagatam/augexp/3302864923__1__3__4', dest = '/home/intern_swagatam/from_sahyadri_with_cpctr/my_first_folder/', num = 70):
    l = glob.glob(root + '/*')
    image_paths = []
    for item in l:
        if item[-4:] == '.tif':
            image_paths.append(item)
    flips = [
        transforms.RandomVerticalFlip(p = 1),
        transforms.RandomHorizontalFlip(p = 1)
        ]
    l_t = [
        transforms.RandomChoice(flips),
        transforms.RandomAffine(degrees = 360, shear = None, resample = False, fillcolor = (255,255,255)),
        transforms.RandomAffine(degrees = 0, shear = 30, resample = False, fillcolor = (255,255,255))
        ]
    l_ = powerset(set(l_t))
    l = [] #The list l stores all possible transformations as lists rather than sets. This is useful later on.
    for item in l_:
        if item:
            item = list(item)
            l.append(item)
    num_tr = len(l)
    images = []
    for image_path in image_paths:
    	images.append(Image.open(image_path))
    # im1 = Image.open(image_paths[0])
    # im2 = Image.open(image_paths[1])
    # im3 = Image.open(image_paths[2])
    # im4 = Image.open(image_paths[3])
    # aug_images = []
    for i in range(num):
    	t = []
    	len_images = len(images)
    	for j in range(len_images):
    		t.append(transforms.Compose(random.choice(l)))

    	f = dest + root.split('/')[-1] + 'aug' + str(i+1)
    	os.mkdir(f)
        
    	for j in range(len_images):
        	tr = t[j]
        	im = images[j]
        	(tr(im)).save(f+'/'+str(j+1)+'.tif')
        # t1 = transforms.Compose(random.choice(l))
        # t2 = transforms.Compose(random.choice(l))
        # t3 = transforms.Compose(random.choice(l))
        # t4 = transforms.Compose(random.choice(l))


        
        
        
        # (t1(im1)).save(f+'/1.tif')
        # (t2(im2)).save(f+'/2.tif')
        # (t3(im3)).save(f+'/3.tif')
        # (t4(im4)).save(f+'/4.tif')
        
        # aug_images.append([t1(im1), t2(im2), t3(im3), t4(im4)])
    pass
    # return aug_images

print('Augmenting 0 class images')
l1 = sorted(glob.glob('/home/test1/PCBN_JHU_CLEAN_CROPPED_CLASSIFIED/train/0' + '/*'))
l2 = sorted(glob.glob('/home/test1/PCBN_JHU_CLEAN_CROPPED_CLASSIFIED/train/1' + '/*'))

i = 0

for path1, path2 in zip(l1,l2):
    _augment(root = path1, dest = '/home/test1/PCBN_JHU/train/0/', num = 22)
    print(f'Class 0 Folder done {i+1}/{94}')
    _augment(root = path2, dest = '/home/test1/PCBN_JHU/train/1/', num = 20)
    print(f'Class 1 Folder done {i+1}/{341}')
    i += 1

# print('Augmenting 1 class images')
# l = glob.glob('/home/intern_swagatam/cpctr_25th_june/CPCTR_4/train/1' + '/*')
# for i, path in enumerate(l):
# 	_augment(root = path)
# 	print(f'Folder done {i+1}/{90}')

print('Have your data!')
from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import os
import random
import glob
import cv2
import re

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

import albumentations as A

from module_list import *
from util.generate_heatmap import img_generate_gaussian, img_generate_centerpoint
from util.point_prompt import get_point_prompt_heatmap, heatmap_label2prompt_heatmap,heatmap_label2prompt_heatmap_dn
# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
def transform(image, label, logits=None, point=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True, det_aug = None, mic = False):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)
    if point is not None:
        point = transforms_f.resize(point, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
        if point is not None:
            point = transforms_f.pad(point, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    transform = A.Compose([
            A.CropNonEmptyMaskIfExists(crop_size[0],crop_size[1],ignore_values = [255],p=0.7),  # 指定裁剪后的高度和宽度
            A.RandomCrop(crop_size[0],crop_size[1]),  # 随机裁剪  
        ],additional_targets={'point': 'mask','logit': 'image'}
        )
    
    # 3. 进行图像和掩码的裁剪
    if logits is not None:
        if point is not None:
            transformed = transform(image=np.array(image), mask=np.array(label), point = np.array(point),logit = np.array(logits))
            image = transformed['image']
            label = transformed['mask']
            point = transformed['point']
            logits = transformed['logit']
        else:
            transformed = transform(image=np.array(image), mask=np.array(label),logit = np.array(logits))
            image = transformed['image']
            label = transformed['mask']
            logits = transformed['logit']
    elif point is not None:
        transformed = transform(image=np.array(image), mask=np.array(label), point = np.array(point))
        image = transformed['image']
        label = transformed['mask']
        point = transformed['point']
    else:
        transformed = transform(image=np.array(image), mask=np.array(label))    
        # 获取裁剪后的图像和掩码
        image = transformed['image']
        label = transformed['mask']

    if augmentation:
        if(det_aug == None):
            aug = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.3),
                A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5],p=0.5),
            ],additional_targets={'point': 'mask','logit': 'mask'})
        else:
            aug = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.3),
                A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5],p=0.5),
                A.ElasticTransform(p=0.3),
                # A.Perspective(p=0.3),
            ],additional_targets={'point': 'mask','logit': 'mask'})
        if logits is not None:
            if point is not None:
                transformed = aug(image=np.array(image), mask=np.array(label), point = np.array(point),logit = np.array(logits))
                image = Image.fromarray(transformed['image'])
                label = Image.fromarray(transformed['mask'])
                point = Image.fromarray(transformed['point'])
                logits = Image.fromarray(transformed['logit'])
            else:
                transformed = aug(image=np.array(image), mask=np.array(label),logit = np.array(logits))
                image = Image.fromarray(transformed['image'])
                label = Image.fromarray(transformed['mask'])
                logits = Image.fromarray(transformed['logit'])
        elif point is not None:
            transformed = aug(image=np.array(image), mask=np.array(label), point = np.array(point))
            image = Image.fromarray(transformed['image'])
            label = Image.fromarray(transformed['mask'])
            point = Image.fromarray(transformed['point'])
        else:
            transformed = aug(image=np.array(image), mask=np.array(label))    
            image = Image.fromarray(transformed['image'])
            label = Image.fromarray(transformed['mask'])
        # Random color jitter
        # if torch.rand(1) > 0.5:
        #     color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  #For PyTorch 1.9/TorchVision 0.10 users
        #     image = color_transform(image)

        # if torch.rand(1) > 0.5:
        #     color_transform = transforms
        #     image = color_transform(image)
        # Random Gaussian filter
        # if torch.rand(1) > 0.5:
        #     sigma = random.uniform(0.15, 1.15)
        #     image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        # if torch.rand(1) > 0.5:
        #     image = transforms_f.hflip(image)
        #     label = transforms_f.hflip(label)
        #     if logits is not None:
        #         logits = transforms_f.hflip(logits)
        #     if point is not None:
        #         point = transforms_f.hflip(point)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)
    if point is not None:
        point = torch.from_numpy(np.array(point)).float()
        # point = transforms_f.to_tensor(point)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        if point is not None:
            return image, label, logits, point
        return image, label, logits
    if point is not None:
        return image, label, point
    else:
        return image, label

def transform_pretrain(image, label, logits=None, point=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True, det_aug = None):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)
    if point is not None:
        point = transforms_f.resize(point, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')
        if point is not None:
            point = transforms_f.pad(point, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    transform = A.Compose([
            A.CropNonEmptyMaskIfExists(crop_size[0],crop_size[1],ignore_values = [255]),  # 指定裁剪后的高度和宽度
        ],additional_targets={'point': 'mask','logit': 'image'}
        )
    
    if logits is not None:
        if point is not None:
            transformed = transform(image=np.array(image), mask=np.array(label), point = np.array(point),logit = np.array(logits))
            image = transformed['image']
            label = transformed['mask']
            point = transformed['point']
            logits = transformed['logit']
        else:
            transformed = transform(image=np.array(image), mask=np.array(label),logit = np.array(logits))
            image = transformed['image']
            label = transformed['mask']
            logits = transformed['logit']
    elif point is not None:
        transformed = transform(image=np.array(image), mask=np.array(label), point = np.array(point))
        image = transformed['image']
        label = transformed['mask']
        point = transformed['point']
    else:
        transformed = transform(image=np.array(image), mask=np.array(label))    
        # 获取裁剪后的图像和掩码
        image = transformed['image']
        label = transformed['mask']

    if augmentation:
        if(det_aug == None):
            aug = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.1),
                A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5],p=0.5),
            ],additional_targets={'point': 'mask','logit': 'mask'})
        else:
            aug = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.1),
                A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5],p=0.5),
                A.ElasticTransform(p=0.3),
                A.Perspective(p=0.3),
            ],additional_targets={'point': 'mask','logit': 'mask'})
        if logits is not None:
            if point is not None:
                transformed = aug(image=np.array(image), mask=np.array(label), point = np.array(point),logit = np.array(logits))
                image = Image.fromarray(transformed['image'])
                label = Image.fromarray(transformed['mask'])
                point = Image.fromarray(transformed['point'])
                logits = Image.fromarray(transformed['logit'])
            else:
                transformed = aug(image=np.array(image), mask=np.array(label),logit = np.array(logits))
                image = Image.fromarray(transformed['image'])
                label = Image.fromarray(transformed['mask'])
                logits = Image.fromarray(transformed['logit'])
        elif point is not None:
            transformed = aug(image=np.array(image), mask=np.array(label), point = np.array(point))
            image = Image.fromarray(transformed['image'])
            label = Image.fromarray(transformed['mask'])
            point = Image.fromarray(transformed['point'])
        else:
            transformed = aug(image=np.array(image), mask=np.array(label))    
            image = Image.fromarray(transformed['image'])
            label = Image.fromarray(transformed['mask'])
            
    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)
    if point is not None:
        point = torch.from_numpy(np.array(point)).float()
        # point = transforms_f.to_tensor(point)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        if point is not None:
            return image, label, logits, point
        return image, label, logits
    if point is not None:
        return image, label, point
    else:
        return image, label


def batch_transform(data, label, logits, point, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list, point_list = [], [], [], []
    device = data.device
    if(point is not None):
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil, point_pil = tensor_to_pil(data[k], label[k], logits[k], point[k])
            # aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
            #                                             crop_size=crop_size,
            #                                             scale_size=scale_size,
            #                                             augmentation=apply_augmentation, det_aug = True)
            aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
                                                        crop_size=crop_size,
                                                        scale_size=scale_size,
                                                        augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)
            point_list.append(aug_point.unsqueeze(0))

        data_trans, label_trans, logits_trans, point_trans = \
            torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device), torch.cat(point_list).to(device)
        return data_trans, label_trans, logits_trans, point_trans
    else:
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
            aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                        crop_size=crop_size,
                                                        scale_size=scale_size,
                                                        augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)

        data_trans, label_trans, logits_trans = \
            torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
        return data_trans, label_trans, logits_trans
    
def batch_transform_mic(data, label, logits, point, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list, point_list = [], [], [], []
    device = data.device
    if(point is not None):
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil, point_pil = tensor_to_pil(data[k], label[k], logits[k], point[k])
            aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
                                                        crop_size=crop_size,
                                                        scale_size=scale_size,
                                                        augmentation=apply_augmentation, mic = True)
            # aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
            #                                             crop_size=crop_size,
            #                                             scale_size=scale_size,
            #                                             augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)
            point_list.append(aug_point.unsqueeze(0))

        data_trans, label_trans, logits_trans, point_trans = \
            torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device), torch.cat(point_list).to(device)
        return data_trans, label_trans, logits_trans, point_trans
    else:
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
            aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                        crop_size=crop_size,
                                                        scale_size=scale_size,
                                                        augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)

        data_trans, label_trans, logits_trans = \
            torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
        return data_trans, label_trans, logits_trans

def batch_transform_det(data, label, logits, point, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list, point_list = [], [], [], []
    device = data.device
    if(point is not None):
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil, point_pil = tensor_to_pil(data[k], label[k], logits[k], point[k])
            aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
                                                        crop_size=crop_size,
                                                        scale_size=scale_size,
                                                        augmentation=apply_augmentation, det_aug = True)
            # aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
            #                                             crop_size=crop_size,
            #                                             scale_size=scale_size,
            #                                             augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)
            point_list.append(aug_point.unsqueeze(0))

        data_trans, label_trans, logits_trans, point_trans = \
            torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device), torch.cat(point_list).to(device)
        return data_trans, label_trans, logits_trans, point_trans
    else:
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
            aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                        crop_size=crop_size,
                                                        scale_size=scale_size,
                                                        augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)

        data_trans, label_trans, logits_trans = \
            torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
        return data_trans, label_trans, logits_trans
# --------------------------------------------------------------------------------
# Define segmentation label re-mapping
# --------------------------------------------------------------------------------

def k_class_map(mask):
    #255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1 # 线粒体
    mask_map[np.isin(mask, [0])] = 255 # 未确定的
    mask_map[np.isin(mask, [128])] = 0 # 背景
    return mask_map

def cvlab_class_map(mask):
    #255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1 # 线粒体
    mask_map[np.isin(mask, [0])] = 255 # 未确定的
    mask_map[np.isin(mask, [128])] = 0 # 背景
    return mask_map
def R_class_map(mask):
    #255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1 # 线粒体
    mask_map[np.isin(mask, [0])] = 255 # 未确定的
    mask_map[np.isin(mask, [128])] = 0 # 背景
    return mask_map

def sam_class_map(mask):
    #255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1 # 线粒体
    mask_map[np.isin(mask, [0])] = 255 # 未确定的
    return mask_map

def sparse_map(mask):
    #255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1 # 线粒体
    mask_map[np.isin(mask, [0])] = 0 # 未确定的
    mask_map[np.isin(mask, [128])] = 0 # 背景
    return mask_map
def label_class_map(mask):
    #255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1 # 线粒体
    return mask_map
def instance_dilate(label,iter1 = 1,iter2 = 4,kernel_size = (4,4)):
    kernel = np.ones(kernel_size, np.uint8)
    img_1 = cv2.dilate(label,kernel,iterations = iter1)
    img_2 = cv2.dilate(label,kernel,iterations = iter2)
    label[(img_2 == 255) & (img_1 != 255)] = 128
    return label

# --------------------------------------------------------------------------------
# Define indices for labelled, unlabelled training images, and test images
# --------------------------------------------------------------------------------

def get_k_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + '/train/img/*.png')
    else:
        file_list = glob.glob(root + '/test/img/*.png')
    idx_list = [int(file[file.find('0'): file.rfind('.')]) for file in file_list]
    idx_list.sort()

    if train:
        return idx_list, idx_list
    else:
        return idx_list
    
def get_cvlab_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + '/train/img/*.png')
    else:
        file_list = glob.glob(root + '/test/img/*.png')
    idx_list = [int(file[re.search(r'\d',file).span()[0]: file.rfind('.')]) for file in file_list]
    idx_list.sort()

    if train:
        return idx_list, idx_list
    else:
        return idx_list
def get_R_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        # file_list = glob.glob(root + '/train/img_crop_10/*.png')
        file_list = glob.glob(root + '/train/img_crop/*.png')
        # idx_list = [int(file[re.search(r'train\d',file).span()[1]: file.rfind('.')]) for file in file_list]
        idx_list = [int(file[re.search(r'train\d',file).span()[1] -1: file.rfind('.')]) for file in file_list]
    else:
        file_list = glob.glob(root + '/test/img_4096/im*.png')
        idx_list = [int(file[re.search(r'im\d',file).span()[1]: file.rfind('.')]) for file in file_list]
    idx_list.sort()
    if train:
        return idx_list, idx_list
    else:
        return idx_list
    
def get_H_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        # file_list = glob.glob(root + '/train/img_crop_10/*.png')
        file_list = glob.glob(root + '/train/img_crop/*.png')
        idx_list = [int(file[re.search(r'train\d',file).span()[1] -1: file.rfind('.')]) for file in file_list]
    else:
        file_list = glob.glob(root + '/test/img_4096/im*.png')
        idx_list = [int(file[re.search(r'im\d',file).span()[1]: file.rfind('.')]) for file in file_list]
    idx_list.sort()
    if train:
        return idx_list, idx_list
    else:
        return idx_list

def get_R_sparse_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_list = []
        x = glob.glob(root + '/train/lab_crop_10_10%instance_dilate3/*.png')
        for i in x:
            if np.array(Image.open(i)).max() == 255:
                file_list.append(i)
        idx_list = [int(file[file.rindex('/') + 1:file.rfind('.')]) for file in file_list]
        # idx_list = [int(file[re.search(r'train\d',file).span()[1]: file.rfind('.')]) for file in file_list]
    else:
        file_list = glob.glob(root + '/test/img_crop_10/*.png')
        idx_list = [int(file[re.search(r'test\d',file).span()[1]: file.rfind('.')]) for file in file_list]
    idx_list.sort()
    if train:
        return idx_list, idx_list
    else:
        return idx_list
# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None, det = False, heatmap_root = None, det_aug = None, target = False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        self.det = det
        self.heatmap_root = heatmap_root
        self.det_aug = det_aug
        self.target = target

    def __getitem__(self, index):
        
        if self.dataset == 'K++':
            if self.train:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == None:
                    label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(np.array(label_root))
                else:
                    label_root = Image.open(self.root + '/train/lab_{}/{}.png'.format(self.apply_partial, str(self.idx_list[index])))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(k_class_map(np.array(label_root)))
                if self.heatmap_root == None:
                    heatmap_root = img_generate_gaussian(label_root, (61,61))
                else:
                    heatmap_root = Image.open(self.heatmap_root + '/{}.png'.format(str(self.idx_list[index])))
                image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation)
                return image, label.squeeze(0), heatmap
            elif self.det:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(np.array(label_root))
            else:
                image_root = Image.open(self.root + '/test/img/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/test/lab/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(np.array(label_root))
            image, label = transform(image_root, label_root, None, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

        if self.dataset == 'cvlab':
            if self.train:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == None:
                    label_root = Image.open(self.root + '/train/gaclab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                    # label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    if(self.target):
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_map = img_generate_centerpoint(label_root)
                else: 
                    label_root = Image.open(self.root + '/train/lab_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(cvlab_class_map(np.array(label_root)))
                    point_map = img_generate_centerpoint(label_root)
                heatmap, _ = img_generate_gaussian(label_root, (61,61))

                image, label, heatmap, point_map = transform(image_root, label_root, heatmap, point_map, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
                promt_point, point_map, instance_label = heatmap_label2prompt_heatmap(point_map, label, drop = True)
                # promt_point, heatmap = get_point_prompt_heatmap(heatmap, drop=True)
                return image, label.squeeze(0), heatmap, promt_point, instance_label.squeeze(0)
                # return image, label.squeeze(0), heatmap, promt_point
            elif self.det:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(np.array(label_root))
            else:
                image_root = Image.open(self.root + '/test/img/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/test/lab/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                heatmap_root = img_generate_centerpoint(label_root)
            image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
            return image, label.squeeze(0), heatmap
            
        if self.dataset == 'R':
            if self.train:
                image_root = Image.open(self.root + '/train/img_crop/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                # image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == None:
                    label_root = Image.open(self.root + '/train/lab_crop/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    # label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    if(self.target):
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_map = img_generate_centerpoint(label_root)
                else:
                    label_root = Image.open(self.root + '/train/lab_crop_{}/train{}.png'.format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(R_class_map(np.array(label_root)))
                    # heatmap_root = Image.fromarray(np.load('heatmap_label/R/1%_2/train' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                    # heatmap_root = img_generate_centerpoint(Image.fromarray(label_class_map(np.array(Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))))))
                    point_map = img_generate_centerpoint(label_root)
                heatmap, _ = img_generate_gaussian(label_root, (61,61))
                image, label, heatmap, point_map = transform(image_root, label_root, heatmap, point_map, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
                promt_point, point_map, instance_label = heatmap_label2prompt_heatmap(point_map, label, drop = True)
                # promt_point, heatmap = get_point_prompt_heatmap(heatmap, drop=True)
                return image, label.squeeze(0), heatmap, promt_point, instance_label.squeeze(0)
                # return image, label.squeeze(0), heatmap, promt_point
            elif self.det:
                image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    sparse_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                else:
                    sparse_root = Image.open(self.root + '/train/lab_crop_10_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))

                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                # sparse_root = Image.fromarray(label_class_map(np.array(sparse_root)))
                heatmap_root, scale = img_generate_gaussian(label_root, (61,61))
                image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)

                return image, label.squeeze(0), heatmap, scale
            else:
                image_root = Image.open(self.root + '/test/img_4096/im{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/test/lab_4096/im{}.png'.format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                # heatmap_root = Image.fromarray(np.load('heatmap_label/R/1%_2/test' + '/{}.npy'.format(str(self.idx_list[index])))[0])

                heatmap_root = img_generate_centerpoint(label_root)
            image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
            return image, label.squeeze(0), heatmap
        
        if self.dataset == 'H':
            if self.train:
                # image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.open(self.root + '/train/img_crop/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == None:
                    # label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    label_root = Image.open(self.root + '/train/lab_crop/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    if(self.target):
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_map = img_generate_centerpoint(label_root)
                else:
                    label_root = Image.open(self.root + '/train/lab_crop_{}/train{}.png'.format(self.apply_partial,str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(R_class_map(np.array(label_root)))
                    # heatmap_root = Image.fromarray(np.load('heatmap_label/H/1%_2/train' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                    point_map = img_generate_centerpoint(label_root)
                heatmap, _ = img_generate_gaussian(label_root, (61,61))
                image, label, heatmap, point_map = transform(image_root, label_root, heatmap, point_map, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
                promt_point, point_map, instance_label = heatmap_label2prompt_heatmap(point_map, label, drop = True)
                # promt_point, heatmap = get_point_prompt_heatmap(heatmap, drop=True)
                return image, label.squeeze(0), heatmap, promt_point, instance_label.squeeze(0)
                # return image, label.squeeze(0), heatmap, promt_point
            elif self.det:
                image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    sparse_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                else:
                    sparse_root = Image.open(self.root + '/train/lab_crop_10_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))

                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                # sparse_root = Image.fromarray(label_class_map(np.array(sparse_root)))
                heatmap_root, scale = img_generate_gaussian(label_root, (61,61))
                image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)

                return image, label.squeeze(0), heatmap, scale
            else:
                image_root = Image.open(self.root + '/test/img_4096/im{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/test/lab_4096/im{}.png'.format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                # heatmap_root = Image.fromarray(np.load('heatmap_label/H/1%_2/test' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                # heatmap_root, scale = img_generate_gaussian(label_root, (61,61))

                heatmap_root = img_generate_centerpoint(label_root)
            image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
            return image, label.squeeze(0), heatmap
    def __len__(self):
        return len(self.idx_list)

class BuildDataset_pretrain_det(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None, det = False, heatmap_root = None, det_aug = None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        self.det = det
        self.heatmap_root = heatmap_root
        self.det_aug = det_aug

    def __getitem__(self, index):
        
        if self.dataset == 'R':
            if self.train:
                image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                else:
                    label_root = Image.open(self.root + '/train/lab_crop_10_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))
                    # label_root = Image.open('pesudo_label_50%/{}.png'.format( self.idx_list[index]))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(R_class_map(np.array(label_root)))
                if self.heatmap_root == None:
                    heatmap_root, _ = img_generate_gaussian(label_root, (61,61),False)

                image, label, heatmap= transform_pretrain(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
                return image, label.squeeze(0), heatmap
            elif self.det: 
                image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    sparse_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                else:
                    sparse_root = Image.open(self.root + '/train/lab_crop_10_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))
                # heatmap_root = np.load(self.root + '/train/heatmap_crop_10/{}.npy'.format( self.idx_list[index]))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                sparse_root = Image.fromarray(label_class_map(np.array(sparse_root)))
                heatmap_root, scale = img_generate_gaussian(label_root, (61,61))
                # scale = np.max(heatmap_root[heatmap_root>0]).astype(np.double)
                # heatmap_root = Image.fromarray(heatmap_root)
                image, label, heatmap = transform_pretrain(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)

                return image, label.squeeze(0), heatmap, scale
            else:
                image_root = Image.open(self.root + '/test/img_crop_10/test{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/test/lab_crop_10/test{}.png'.format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                heatmap_root = Image.open('pesudo_heatmap_test' + '/{}.png'.format(str(self.idx_list[index])))
            image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
            return image, label.squeeze(0), heatmap
    def __len__(self):
        return len(self.idx_list)


class BuildDataset_pretrain(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None, det = False, heatmap_root = None, det_aug = None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        self.det = det
        self.heatmap_root = heatmap_root
        self.det_aug = det_aug

    def __getitem__(self, index):
        
        if self.dataset == 'K++':
            if self.train:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == None:
                    label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                else:
                    label_root = Image.open(self.root + '/train/lab_{}/train{}.png'.format(self.apply_partial, str(self.idx_list[index]).zfill(3)))

                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(k_class_map(np.array(label_root)))
                if self.heatmap_root == None:
                    heatmap_root = img_generate_gaussian(label_root, (61,61))
                else:
                    heatmap_root = Image.open(self.heatmap_root + '/{}.png'.format(str(self.idx_list[index])))
                image, label, heatmap = transform_pretrain(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation)
                return image, label.squeeze(0), heatmap
            elif self.det:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(np.array(label_root))
            else:
                image_root = Image.open(self.root + '/test/img/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/test/lab/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(np.array(label_root))
            image, label = transform_pretrain(image_root, label_root, None, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

        if self.dataset == 'cvlab':
            if self.train:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                else:
                    label_root = Image.open(self.root + '/train/lab_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(cvlab_class_map(np.array(label_root)))
                if self.heatmap_root == None:
                    heatmap_root = img_generate_gaussian(label_root, (61,61))
                else:
                    heatmap_root = Image.open(self.heatmap_root + '/{}.png'.format(str(self.idx_list[index])))
                image, label, heatmap = transform_pretrain(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation)
                return image, label.squeeze(0), heatmap
            elif self.det:
                image_root = Image.open(self.root + '/train/img/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/train/lab/train{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(np.array(label_root))
            else:
                image_root = Image.open(self.root + '/test/img/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + '/test/lab/test{}.png'.format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
            image, label = transform_pretrain(image_root, label_root, None, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)
        if self.dataset == 'R':
            if self.train:
                image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                else:
                    label_root = Image.open(self.root + '/train/lab_crop_10_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))
                    # label_root = Image.open('pesudo_label_50%/{}.png'.format( self.idx_list[index]))
                    image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                    label_root = R_class_map(np.array(label_root))
                if self.heatmap_root == None:
                    # heatmap_root, _ = img_generate_gaussian(label_root, (61,61),False)
                    heatmap_root = np.array(Image.open('heatmap_label/pesudo_heatmap_50%/{}.png'.format(self.idx_list[index])))

                    # heatmap_root[label_root == 1] = 0
                    # label_root[heatmap_root == 255] = 1
                    heatmap_root = Image.fromarray(heatmap_root)
                    label_root = Image.fromarray(label_root)
                image, label, heatmap= transform_pretrain(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
                return image, label.squeeze(0), heatmap
            elif self.det: 
                image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    sparse_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                else:
                    sparse_root = Image.open(self.root + '/train/lab_crop_10_{}/{}.png'.format(self.apply_partial, self.idx_list[index]))
                # heatmap_root = np.load(self.root + '/train/heatmap_crop_10/{}.npy'.format( self.idx_list[index]))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                sparse_root = Image.fromarray(label_class_map(np.array(sparse_root)))
                heatmap_root, scale = img_generate_gaussian(label_root, (61,61))
                # scale = np.max(heatmap_root[heatmap_root>0]).astype(np.double)
                # heatmap_root = Image.fromarray(heatmap_root)
                image, label, heatmap = transform_pretrain(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
                # box = extract_boxs(sparse.unsqueeze(0), 3)
                # box_path = os.path.join(self.root + '/train', 'box_example_coordinates/' + str(self.idx_list[index])+'.txt')
                # bboxes_list = []
                # with open(box_path, 'r') as f:
                #     for line in f:
                #         # 解析每行数据
                #         values = list(map(float, line.strip()[1:-1].split(',')))
                #         bboxes_list.append(values)
                # bboxes = np.array(bboxes_list).astype(np.float32)
                # bboxes = torch.from_numpy(bboxes)/ torch.tensor([1024, 1024, 1024, 1024]) * 512
                return image, label.squeeze(0), heatmap, scale
            else:
                image_root = Image.open(self.root + '/test/img_crop_10/test{}.png'.format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + '/test/lab_crop_10/test{}.png'.format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.fromarray(np.stack((image_root,image_root,image_root),axis = -1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                heatmap_root = Image.open('heatmap_label/pesudo_heatmap_test' + '/{}.png'.format(str(self.idx_list[index])))
            image, label, heatmap = transform(image_root, label_root, None, heatmap_root, self.crop_size, self.scale_size, self.augmentation, self.det_aug)
            return image, label.squeeze(0), heatmap
    def __len__(self):
        return len(self.idx_list)
# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataLoader:
    def __init__(self, dataset,crop_size,batch_size,target = False):
        self.dataset = dataset

        if dataset == 'K++':
            self.data_path = 'dataset/K++'
            self.im_size = [1463, 1613]
            self.test_size = [1334, 1553]
            self.crop_size = [512, 512]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_k_idx(self.data_path, train=True)
            self.test_idx = get_k_idx(self.data_path, train=False)
        if dataset == 'cvlab':
            self.data_path = 'dataset/cvlab'
            self.im_size = [768, 1024]
            self.test_size = [768, 1024]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_cvlab_idx(self.data_path, train=True)
            self.test_idx = get_cvlab_idx(self.data_path, train=False)
            self.target = target
        if dataset == 'R':
            self.data_path = 'dataset/R'
            self.im_size = [1024, 1024]
            self.test_size = [4096, 4096]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_R_idx(self.data_path, train=True)
            self.test_idx = get_R_idx(self.data_path, train=False)
            self.target = target
        if dataset == 'H':
            self.data_path = 'dataset/H'
            self.im_size = [1024, 1024]
            self.test_size = [4096, 4096]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_H_idx(self.data_path, train=True)
            self.test_idx = get_H_idx(self.data_path, train=False)
            self.target = target

    def build(self, partial=None, partial_seed=None, det_aug=None, num_samples = None):
        train_l_dataset = BuildDataset(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=False, train=True, apply_partial=partial, partial_seed=partial_seed, det_aug = det_aug, target = self.target)

        test_dataset    = BuildDataset(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.test_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        if(num_samples == None):
            num_samples = len(train_l_dataset)
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=False,
                                          num_samples=num_samples),
            drop_last=True,
            pin_memory = True,
            num_workers = 8
        )


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory = True,
            num_workers = 4
        )
        
        return train_l_loader, test_loader

    def build_det(self, partial=None, partial_seed=None, det_aug = None):
        train_l_dataset = BuildDataset(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=False, train=True, apply_partial=partial, partial_seed=partial_seed, target=self.target)

        test_dataset    = BuildDataset(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.test_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        test_det_dataset    = BuildDataset(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),apply_partial=partial,
                                       augmentation=False, train=False,det=True)
        
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=False,
                                          num_samples=len(train_l_dataset)),
            drop_last=True,
            pin_memory = True,
            num_workers = 8
        )


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory = True,
            num_workers = 8
        )
        
        test_det_loader = torch.utils.data.DataLoader(
            test_det_dataset,
            batch_size=4,
            shuffle=False,
            pin_memory = True,
            num_workers = 4
        )
        return train_l_loader, test_loader, test_det_loader

    def build_pretrain(self, partial=None, partial_seed=None,det_aug = None):
        train_l_dataset = BuildDataset_pretrain(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed, det_aug = det_aug)

        test_dataset    = BuildDataset_pretrain(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.test_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        test_det_dataset    = BuildDataset_pretrain(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),apply_partial=partial,
                                       augmentation=False, train=False,det=True)
        
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=2,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=False,
                                          num_samples=len(train_l_dataset)),
            drop_last=True,
            pin_memory = True,
            num_workers = 8
        )


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            pin_memory = True,
            num_workers = 8
        )
        
        test_det_loader = torch.utils.data.DataLoader(
            test_det_dataset,
            batch_size=4,
            shuffle=False,
            pin_memory = True,
            num_workers = 8
        )
        return train_l_loader, test_loader, test_det_loader

    def build_pretrain_det(self, partial=None, partial_seed=None,det_aug = None):
        train_l_dataset = BuildDataset_pretrain_det(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed, det_aug = det_aug)

        test_dataset    = BuildDataset_pretrain_det(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.test_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        test_det_dataset    = BuildDataset_pretrain_det(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),apply_partial=partial,
                                       augmentation=False, train=False,det=True)
        
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=4,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=False,
                                          num_samples=len(train_l_dataset)),
            drop_last=True,
            pin_memory = True,
            num_workers = 8
        )


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory = True,
            num_workers = 8
        )
        
        test_det_loader = torch.utils.data.DataLoader(
            test_det_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory = True,
            num_workers = 8
        )
        return train_l_loader, test_loader, test_det_loader
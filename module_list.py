import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import random
import cv2
import monai
import torch.nn as nn
from PIL import Image

from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms.functional as transforms_f
from util.generate_heatmap import generate_gaussian

# --------------------------------------------------------------------------------
# Define EMA: Mean Teacher Framework
# --------------------------------------------------------------------------------
class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        
    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

#fixed ema
class EMA_fixed(object):
    def __init__(self, model, alpha):
        self.model = copy.deepcopy(model)
        self.alpha = alpha
            
    def update(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data =self.alpha * ema_param.data + (1 - self.alpha) * param.data

# --------------------------------------------------------------------------------
# Define Polynomial Decay
# --------------------------------------------------------------------------------
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

# --------------------------------------------------------------------------------
# Define training losses
# --------------------------------------------------------------------------------
def compute_supervised_loss(predict, target, reduction=True):
    # target[~mask_gt] = -1
    if reduction:
        loss = F.binary_cross_entropy(predict[:,1,::], target.float())
    else:
        loss = F.binary_cross_entropy(predict[:,1,::], target.float(), reduction='none')
    return loss

def compute_supervised_loss_ignore(predict, target, ignore_index = -1):
    mask = target != ignore_index
    target[target == ignore_index] = 0
    loss = F.binary_cross_entropy(predict[:,1,::], target.float(), reduction='none')
    loss = torch.sum(loss * mask)/torch.sum(mask)
    return loss

def compute_unsupervised_loss(predict, target, logits, mask_gt, strong_threshold, weak_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels
    # target[mask_gt] = -1
    # target[(target == 1) * (logits < weak_threshold)] = -1
    # target[(target == 0) * (logits < 0.6)] = -1
    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    # print('weighting{}'.format(weighting))
    loss = F.binary_cross_entropy(predict[:,1,::], target.float(), reduction='none')
    loss[(target == 1) * (logits < weak_threshold)] = 0
    loss[(target == 0) * (logits < weak_threshold)] = 0
    # loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss

# --------------------------------------------------------------------------------
# Define evaluation metrics
# --------------------------------------------------------------------------------
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.jac = 0
        self.dice = 0
        self.num = 0
        self.compute_dice = monai.metrics.DiceMetric(include_background=False,ignore_empty=False)

    def update(self, pred, target):
        binary = torch.nn.functional.one_hot(target,num_classes=2).permute(0,3,1,2)
        pre = torch.nn.functional.one_hot(pred.argmax(1),num_classes=2).permute(0,3,1,2)
        jac = monai.metrics.compute_iou(pre,binary, include_background=False,ignore_empty=False)
        dice = self.compute_dice(pre,binary)
        self.jac += torch.mean(jac)
        self.dice += torch.mean(dice)
        self.num += 1

    def get_jac(self):
        return self.jac/self.num
    
    def get_dice(self):
        return self.dice/self.num
# --------------------------------------------------------------------------------
# Define useful functions
# --------------------------------------------------------------------------------
def label_binariser(inputs):
    outputs = torch.zeros_like(inputs).to(inputs.device)
    index = torch.max(inputs, dim=1)[1]
    outputs = outputs.scatter_(1, index.unsqueeze(1), 1.0)
    return outputs


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def tensor_to_pil(im, label, logits, point = None):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    if(point is not None):  
        point = transforms_f.to_pil_image(point.unsqueeze(0).cpu())
        return im, label, logits, point
    else:
        return im, label, logits

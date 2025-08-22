import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2  # pad = 1  //整除

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)  # 这里使用max_pooling 来简化计算
    keep = (hmax == heat).float()  # 找到最大值们相对位置，位置相同为True不同为False
    return heat * keep


# nms find center point
def genarate_point(pseudo):
    """Filter pseudo-words from a list of words."""
    point = []

    for j in range(len(pseudo)):
        label = pseudo[j].detach().cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(label)
        gt = np.zeros(label.shape[:2])
        for j in range(1, num_labels):
            # 获取当前连通域的像素坐标
            mask = labels == j
            # 计算连通域的质心
            M = cv2.moments(mask.astype(np.uint8))
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            gt[cY, cX] = 1
        point.append(gt)

    return torch.from_numpy(np.array(point)).to("cuda:0")
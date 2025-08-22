import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import glob
import re
from scipy.ndimage import center_of_mass


def generate_gaussian(foregrounds, size):

    heatmap = []
    for j in range(len(foregrounds)):
        label = foregrounds[j].cpu().numpy().astype(np.uint8)
        retval, connections = cv2.connectedComponents(label)
        dist_img = cv2.distanceTransform(label, cv2.DIST_L2, 3)

        gt = np.zeros(dist_img.shape[:2])
        for i in range(1, retval):
            mask = connections == i
            # if(istrain):
            #     if(mask.sum() < 64 ):#排除掉一些很小的pixel
            #         continue
            # 计算连通域的质心
            M = cv2.moments(mask.astype(np.uint8))
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            gt[cY, cX] = 1

        det_map = np.zeros_like(label, dtype=np.uint8)
        if np.count_nonzero(gt) != 0:
            det_map = cv2.GaussianBlur(gt, size, 0, borderType=0)
            # det_map = gaussian_filter(gt, 10, mode='constant')
            am = np.min(det_map[gt > 0]).astype(np.double)
            if am != 0 and det_map.sum() != 0:
                det_map /= am
                det_map[det_map > 1] = 1
                det_map = det_map * 255.0
                #     heatmap.append(det_map.astype(float))
                # else:
                #     det_map = det_map
                #     heatmap.append(det_map.astype(float))
                heatmap.append(det_map.astype(float))
        else:
            heatmap.append(det_map.astype(float))
    return torch.from_numpy(np.array(heatmap)).cuda()


def img_generate_gaussian(foregrounds, size, istrain=True):
    """
    0背景，1前景，255未知转化为只有0和1前景
    """
    label = np.array(foregrounds)
    if np.unique(label).shape == (3,):
        if all(np.unique(label) == [0, 1, 255]):  # 0背景，1前景，255未知转化为只有0和1前景
            label[label == 255] = 0
    if np.unique(label).shape == (2,):
        # if all(np.unique(label) == [0, 255]):  # 0背景，1前景，255未知转化为只有0和1前景
        #     label[label == 255] = 0
        if all(np.unique(label) == [1, 255]):  # 0背景，1前景，255未知转化为只有0和1前景
            label[label == 255] = 0

    retval, connections = cv2.connectedComponents(label)
    dist_img = cv2.distanceTransform(label, cv2.DIST_L2, 3)

    gt = np.zeros(dist_img.shape[:2])
    if len(np.unique(connections)) == 1:
        retval = 1
    for j in range(1, retval):
        mask = connections == j
        # if(istrain):
        #     if(mask.sum() < 64 ):#排除掉一些很小的pixel
        #         continue
        # 计算连通域的质心
        M = cv2.moments(mask.astype(np.uint8))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        gt[cY, cX] = 1

    det_map = np.zeros_like(label, dtype=np.uint8)
    am = 0
    if np.count_nonzero(gt) != 0:
        det_map = cv2.GaussianBlur(gt, size, 0, borderType=0)
        # det_map = gaussian_filter(gt,10, mode='constant')
        am = np.min(det_map[gt > 0]).astype(np.double)
        if am != 0 and det_map.sum() != 0:
            det_map /= am
            det_map[det_map > 1] = 1
            det_map = det_map * 255.0
    return Image.fromarray(det_map), am

def is_inside(binary_mask, point):
    """
    判断point是否在mask中为1的区域内。
    """
    x, y = point
    h, w = binary_mask.shape
    if 0 <= y < h and 0 <= x < w:
        return binary_mask[y, x] > 0
    return False

def img_generate_centerpoint(foregrounds):
    """
    输入：foregrounds - 图像，像素值为 0（背景）、1（前景）、255（未知）
    输出：每个前景区域标出中心点的图像（255为中心点）
    """
    label = np.array(foregrounds)
    
    # 统一将未知区域转为背景
    unique_vals = np.unique(label)
    if np.array_equal(unique_vals, [0, 1, 255]):
        label[label == 255] = 0
    elif np.array_equal(unique_vals, [1, 255]):
        label[label == 255] = 0

    # 查找连通区域
    num_labels, labels = cv2.connectedComponents(label.astype(np.uint8))
    
    # 初始化输出图像
    gt = np.zeros(label.shape[:2], dtype=np.uint8)

    centroids = []

    for i in range(1, num_labels):  # 忽略背景 label=0
        comp_mask = (labels == i).astype(np.uint8)

        # Step 1: 几何质心
        M = cv2.moments(comp_mask)
        if M["m00"] != 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
            if is_inside(comp_mask, (cx, cy)):
                centroids.append((cx, cy))
                gt[cy, cx] = 255
                continue

        # Step 2: 像素重心
        cy_cm, cx_cm = center_of_mass(comp_mask)
        cx = int(round(cx_cm))
        cy = int(round(cy_cm))
        if is_inside(comp_mask, (cx, cy)):
            centroids.append((cx, cy))
            gt[cy, cx] = 255
            continue

        # Step 3: fallback - 找最近像素
        coords = np.column_stack(np.where(comp_mask > 0))
        distances = np.linalg.norm(coords - np.array([cy_cm, cx_cm]), axis=1)
        cy_final, cx_final = coords[np.argmin(distances)]
        centroids.append((int(cx_final), int(cy_final)))
        gt[cy_final, cx_final] = 255

    return Image.fromarray(gt)



def img_generate_gaussian_(foregrounds, size):
    """
    0未知,128背景,255前景转化为只有0和255前景
    """
    label = np.array(foregrounds)
    label[label == 128] = 0
    retval, connections = cv2.connectedComponents(label)
    dist_img = cv2.distanceTransform(label, cv2.DIST_L2, 3)

    gt = np.zeros(dist_img.shape[:2])
    for j in range(1, retval):
        mask = connections == j
        if mask.sum() < 64:  # 排除掉一些很小的pixel
            continue
        # 计算连通域的质心
        M = cv2.moments(mask.astype(np.uint8))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        gt[cY, cX] = 1

    det_map = np.zeros_like(label, dtype=np.uint8)
    if np.count_nonzero(gt) != 0:
        det_map = cv2.GaussianBlur(gt, size, 0, borderType=0)
        am = np.min(det_map[gt > 0]).astype(np.double)
        if am != 0 and det_map.sum() != 0:
            det_map /= am
            det_map[det_map > 1] = 1
            det_map = det_map * 255.0
    return Image.fromarray(det_map.astype(np.uint8))


if __name__ == "__main__":

    label_list = glob.glob("dataset/R/train/lab_crop_10/*.png")
    for label_path in label_list:
        i = int(label_path[re.search(r"train\d", label_path).span()[1] : label_path.rfind(".")])
        foregrounds = Image.open(label_path)
        label = np.array(foregrounds)
        label[label == 128] = 0
        label[label == 255] = 1
        retval, connections = cv2.connectedComponents(label)
        dist_img = cv2.distanceTransform(label, cv2.DIST_L2, 3)

        gt = np.zeros(dist_img.shape[:2])
        for j in range(1, retval):
            mask = connections == j
            # 计算连通域的质心
            M = cv2.moments(mask.astype(np.uint8))
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            gt[cY, cX] = 1

        det_map = gaussian_filter(gt, 10, mode="constant")
        np.save("dataset/R/train/heatmap_crop_10/{}.npy".format(i), det_map)

from util.pseudo_filter import nms
import torch
import numpy as np
import cv2


def get_point_prompt(heatmap, threshold=125, drop=False):
    """get point prompt from heatmap

    :param heatmap: torch.tensor H*W
    """
    heatmap[heatmap < threshold] = 0
    # heatmap[heatmap < heatmap.max()*0.5] = 0
    filter_p = nms(heatmap.unsqueeze(0), kernel=5)[0]

    # 获取非0元素及其位置
    non_zero_indices = torch.nonzero(filter_p, as_tuple=False)[:, [1, 0]]  # 获取非 0 元素的二维索引
    non_zero_elements = filter_p[non_zero_indices[:, 1], non_zero_indices[:, 0]]  # 获取非0元素

    # 对非0元素进行排序
    sorted_elements, sorted_indices = torch.sort(non_zero_elements)

    # 获取排序后元素对应的原始位置
    sorted_positions = non_zero_indices[sorted_indices]
    if drop:
        indices = torch.randperm(sorted_positions.shape[0])[: np.random.randint(0, sorted_positions.shape[0] + 1)]
        sorted_positions = sorted_positions[indices]
    # print(len(sorted_positions))
    return torch.cat([sorted_positions, -torch.ones([100 - sorted_positions.shape[0], 2], device=heatmap.device)])

def get_point_prompt_heatmap(heatmap, threshold=125, drop=False):
    """get point prompt from heatmap

    :param heatmap: torch.tensor H*W
    """
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap < heatmap.max() * 0.5] = 0
    filter_p = nms(heatmap.unsqueeze(0), kernel=3)[0]

    # 获取非0元素及其位置
    non_zero_indices = torch.nonzero(filter_p, as_tuple=False)[:, [1, 0]]  # 获取非 0 元素的二维索引
    non_zero_elements = filter_p[non_zero_indices[:, 1], non_zero_indices[:, 0]]  # 获取非0元素

    # 对非0元素进行排序
    sorted_elements, sorted_indices = torch.sort(non_zero_elements)

    # 获取排序后元素对应的原始位置
    sorted_positions = non_zero_indices[sorted_indices]
    if drop:
        indices = torch.randperm(sorted_positions.shape[0])[: np.random.randint(0, sorted_positions.shape[0] + 1)]
        sorted_positions = sorted_positions[indices]
    # print(len(sorted_positions))
    return torch.cat([sorted_positions, -torch.ones([10 - sorted_positions.shape[0], 2])]), filter_p


def heatmap_label2prompt_heatmap(pointmap, label, threshold=125, drop=False):
    """get point prompt from heatmap

    :param heatmap: torch.tensor H*W
    """
    lab = label.cpu().numpy().copy()
    lab[lab == -1] = 0

    pointmap[pointmap < threshold] = 0
    pointmap[pointmap < pointmap.max() * 0.5] = 0
    filter_p = nms(pointmap.unsqueeze(0), kernel=3)[0]

    # 获取非0元素及其位置
    non_zero_indices = torch.nonzero(filter_p, as_tuple=False)[:, [1, 0]]  # 获取非 0 元素的二维索引
    non_zero_elements = filter_p[non_zero_indices[:, 1], non_zero_indices[:, 0]]  # 获取非0元素

    # 对非0元素进行排序
    sorted_elements, sorted_indices = torch.sort(non_zero_elements)

    # 获取排序后元素对应的原始位置
    sorted_positions = non_zero_indices[sorted_indices]
    if drop:
        indices = torch.randperm(sorted_positions.shape[0])[: np.random.randint(0, sorted_positions.shape[0] + 1)]
        sorted_positions = sorted_positions[indices]
    # print(len(sorted_positions))
    mask_partial = torch.zeros_like(label)
    pointmap_partial = torch.zeros_like(filter_p)
    instance_mask = lab.astype(np.uint8)[0]

    # 使用cv2.connectedComponents找到连通区域
    num_labels, labeled_mask = cv2.connectedComponents(instance_mask, connectivity=4)

    for label_id in np.atleast_1d(labeled_mask[sorted_positions[:, 1], sorted_positions[:, 0]]):  # 跳过背景标签0
        # if label_id == 0:
        #     continue
        assert label_id != 0,"label_id should not be 0"
        mask_partial[0, labeled_mask == label_id] = 1
    for x, y in sorted_positions:
        if x < 0 or y < 0 or x >= pointmap.shape[1] or y >= pointmap.shape[0]:
            continue
        pointmap_partial[y, x] = 255
    return (
        torch.cat([sorted_positions, -torch.ones([50 - sorted_positions.shape[0], 2])]),
        filter_p,
        mask_partial,
        pointmap_partial,
    )

def heatmap_label2prompt_heatmap_dn(heatmap, label, threshold=125, drop=False):
    """get point prompt from heatmap

    :param heatmap: torch.tensor H*W
    """
    lab = label.cpu().numpy()

    heatmap[heatmap < threshold] = 0
    heatmap[heatmap < heatmap.max() * 0.5] = 0
    filter_p = nms(heatmap.unsqueeze(0), kernel=3)[0]

    # 获取非0元素及其位置
    non_zero_indices = torch.nonzero(filter_p, as_tuple=False)[:, [1, 0]]  # 获取非 0 元素的二维索引
    non_zero_elements = filter_p[non_zero_indices[:, 1], non_zero_indices[:, 0]]  # 获取非0元素

    # 对非0元素进行排序
    sorted_elements, sorted_indices = torch.sort(non_zero_elements)

    # 获取排序后元素对应的原始位置
    sorted_positions = non_zero_indices[sorted_indices]
    if drop:
        indices = torch.randperm(sorted_positions.shape[0])[: np.random.randint(0, sorted_positions.shape[0] + 1)]
        sorted_drop_positions = sorted_positions[indices]
    # print(len(sorted_positions))
    prompt_mask = -torch.ones_like(label)
    instance_mask = lab.astype(np.uint8)[0]

    # 使用cv2.connectedComponents找到连通区域
    num_labels, labeled_mask = cv2.connectedComponents(instance_mask, connectivity=4)

    for label_id in np.atleast_1d(labeled_mask[sorted_drop_positions[:, 1], sorted_drop_positions[:, 0]]):  # 跳过背景标签0
        prompt_mask[0, labeled_mask == label_id] = 1

    # n组加噪实例中心点（n*点数*2）和对应的mask（点数*H*W）
    dn_point = -torch.ones([5, 50, 2])
    dn_mask = torch.zeros([30, prompt_mask.shape[1], prompt_mask.shape[2]])
    dn_background_mask = torch.tensor(instance_mask == 0)
    background = instance_mask == 0
    background_position = np.argwhere(background)[:, [1, 0]]
    if num_labels != 1:
        for j in range(1, num_labels):
            mask = labeled_mask == j
            position = np.argwhere(mask)[:, [1, 0]]
            for i in range(5):
                dn_point[i, j - 1] = torch.tensor(position[np.random.randint(0, position.shape[0])])
            dn_mask[j - 1, ::] = torch.tensor(mask)
        for i in range(5):
            for j in range(num_labels - 1, num_labels * 2 - 1):
                dn_point[i, j] = torch.tensor(background_position[np.random.randint(0, background_position.shape[0])])
    return (
        torch.cat([sorted_drop_positions, -torch.ones([50 - sorted_drop_positions.shape[0], 2])]),
        filter_p,
        prompt_mask,
        dn_point,
        dn_mask,
    )

def get_point_prompt_batch(heatmap, threshold=125, drop=False):
    """get point prompt from heatmap

    :param heatmap: torch.tensor B*H*W
    """
    point_prompt = []
    for h in heatmap:
        point_prompt.append(get_point_prompt(h.cpu(), threshold, drop=drop))
    return torch.stack(point_prompt)
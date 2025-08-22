import torch
import torch.nn.functional as F
from util.point_prompt import get_point_prompt
import torchvision
import torchvision.transforms.functional as TF

@torch.no_grad()
def inference_det(img, model, heatmap, window_size, window_stride, batch_size, is_aug=False):
    ori_shape = img.shape[2:]
    output = []
    output_heatmap = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for im, ht in zip(img, heatmap):
        seg_maps = []
        heatmap_maps = []
        anchors = []
        angles = []

        for batch_crops, batch_points, batch_info in sliding_window_batch(im.unsqueeze(0), ht, window_size, window_stride, is_aug, batch_size):
            batch_crops = batch_crops.to(device)
            batch_points = batch_points.to(device)

            pred, pred_heatmap = model(batch_crops.squeeze(1), batch_points)
            pred_up = F.interpolate(pred, window_size, mode="bilinear", align_corners=True).cpu()
            heatmap_up = F.interpolate(pred_heatmap, window_size, mode="bilinear", align_corners=True).cpu()

            seg_maps.append(pred_up)
            heatmap_maps.append(heatmap_up)
            anchors.extend([(ha, wa) for ha, wa, angle in batch_info])
            if is_aug:
                angles.extend([angle for _, _, angle in batch_info])

        seg_maps = torch.cat(seg_maps, dim=0)
        heatmap_maps = torch.cat(heatmap_maps, dim=0)

        windows = {
            "point": None,
            "anchors": anchors,
            "seg_maps": seg_maps,
            "heatmaps": heatmap_maps,
            "shape": ori_shape,
        }
        if is_aug:
            windows["angle"] = angles

        im_seg_map, pred_heatmap = merge_windows_det(windows, window_size, ori_shape, is_aug)
        output.append(im_seg_map)
        output_heatmap.append(pred_heatmap)

    return torch.stack(output), torch.stack(output_heatmap)

def sliding_window_batch(im, ht, window_size, window_stride, is_aug=False, batch_size=16):
    B, C, H, W = im.shape
    ws = window_size
    device = im.device

    h_anchors = torch.arange(0, H, window_stride, device=device)
    w_anchors = torch.arange(0, W, window_stride, device=device)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]

    current_crop_batch = []
    current_point_batch = []
    current_info_batch = []  # 用于返回 anchor, angle 信息
    angles = [0, 90, 180, 270] if is_aug else [0]

    for ha in h_anchors:
        for wa in w_anchors:
            for angle in angles:
                crop = im[:, :, ha:ha + ws, wa:wa + ws]
                crop_rot = TF.rotate(crop, angle)
                ht_patch = ht[ha:ha + ws, wa:wa + ws].unsqueeze(0)
                ht_rot = TF.rotate(ht_patch, angle).squeeze(0)
                point = get_point_prompt(ht_rot)

                current_crop_batch.append(crop_rot)
                current_point_batch.append(point)
                current_info_batch.append((ha, wa, angle))

                if len(current_crop_batch) == batch_size:
                    yield (
                        torch.stack(current_crop_batch),
                        torch.stack(current_point_batch),
                        list(current_info_batch)
                    )
                    current_crop_batch = []
                    current_point_batch = []
                    current_info_batch = []

    # yield 最后一批不足 batch_size 的数据
    if current_crop_batch:
        yield (
            torch.stack(current_crop_batch),
            torch.stack(current_point_batch),
            list(current_info_batch)
        )


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count
    return logit

@torch.no_grad()
def merge_windows_det(windows, window_size, ori_shape, is_aug=False):
    ws = window_size
    seg_maps = windows["seg_maps"]
    heatmap_maps = windows["heatmaps"]
    anchors = windows["anchors"]
    H, W = windows["shape"]
    device = seg_maps.device

    # 使用展开操作合并结果（关键优化点4）
    C = seg_maps.size(1)
    logit = torch.zeros((C, H, W), device=device)
    count = torch.zeros((1, H, W), device=device)
    heatmap = torch.zeros((1, H, W), device=device)

    if is_aug:
        angles = windows["angle"]
        for idx, (ha, wa) in enumerate(anchors):
            angle = angles[idx]
            # 反向旋转预测结果
            seg_rot = torchvision.transforms.functional.rotate(seg_maps[idx].unsqueeze(0), -angle).squeeze(0)
            ht_rot = torchvision.transforms.functional.rotate(heatmap_maps[idx].unsqueeze(0), -angle).squeeze(0)
            # predict[:, ha : ha + ws, wa : wa + ws] += seg_rot[1] > 0.5
            mask = seg_rot[1] > 0.5
            logit[:, ha : ha + ws, wa : wa + ws] += seg_rot
            heatmap[:, ha : ha + ws, wa : wa + ws] += ht_rot
            count[:, ha : ha + ws, wa : wa + ws] += 1
    else:
        # 向量化累加（关键优化点4）
        for idx, (ha, wa) in enumerate(anchors):
            # torch.max(seg_maps[idx], dim=0)[1]
            # predict[:, ha : ha + ws, wa : wa + ws] += seg_maps[idx][1] > 0.5
            mask = seg_maps[idx][1] > 0.5
            logit[:, ha : ha + ws, wa : wa + ws] += seg_maps[idx]
            heatmap[:, ha : ha + ws, wa : wa + ws] += heatmap_maps[idx]
            count[:, ha : ha + ws, wa : wa + ws] += 1

    # 避免除零
    logit = logit / count.clamp(min=1e-6)
    heatmap = heatmap / count.clamp(min=1e-6)
    return logit, heatmap
    # return logit, heatmap

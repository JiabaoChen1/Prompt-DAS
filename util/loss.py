import sys

sys.path.append(".")
import torch
import torch.nn.functional as F
import random
import torch.nn as nn

def contrast_loss(feature, sample_num=3, temp=0.5):
    """simple contrast loss

    :param feature: detection similarity map feature | tuple(postivie feature, negative feature)
    """
    positive_feature, negative_feature = feature
    device = positive_feature.device
    contrast_loss = torch.tensor(0.0)
    # sample queries
    anchor_feat = positive_feature.clone()
    with torch.no_grad():  # sample negative and positve
        negative_feat = negative_feature[range(sample_num)].unsqueeze(0).repeat(3, 1, 1)
        positive_feat = torch.stack(
            [
                positive_feature[random.choice([1, 2])],
                positive_feature[random.choice([0, 2])],
                positive_feature[random.choice([0, 1])],
            ]
        ).unsqueeze(1)

    all_feat = torch.cat((positive_feat, negative_feat), dim=1)
    seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
    contrast_loss = contrast_loss + F.cross_entropy(seg_logits / temp, torch.zeros(3).long().to(device))

    # sample queries
    anchor_feat = negative_feature.clone()
    with torch.no_grad():  # sample negative and positve
        negative_feat = positive_feature[range(sample_num)].unsqueeze(0).repeat(3, 1, 1)
        positive_feat = torch.stack(
            [
                negative_feature[random.choice([1, 2])],
                negative_feature[random.choice([0, 2])],
                negative_feature[random.choice([0, 1])],
            ]
        ).unsqueeze(1)

    all_feat = torch.cat((positive_feat, negative_feat), dim=1)
    seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

    contrast_loss = contrast_loss + F.cross_entropy(seg_logits / temp, torch.zeros(3).long().to(device))

    return contrast_loss


# contrast_loss((torch.zeros(3,256), torch.zeros(3,256)))


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, gt):
        target = target.float()
        smooth = 1e-5

        intersect = torch.sum((score * target)[gt[:, 0, ...] >= 0])
        y_sum = torch.sum((target * target)[gt[:, 0, ...] >= 0])
        z_sum = torch.sum((score * score)[gt[:, 0, ...] >= 0])

        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        gt = target.clone()
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        class_wise_dice = []
        loss = torch.tensor(0.0).to(device=inputs.device)
        for i in range(0, self.n_classes):
            if target[:, i].sum() == 0:
                continue
            dice = self._dice_loss(inputs[:, i], target[:, i], gt)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def token_contrast(feature_list, temperature=0.5):
    """计算token对比损失

    :param feature: torch.tensor
    :param temperature: float
    """
    # loss = torch.tensor(0.0)
    # positive,  query, negative = [], [], []
    # for feature in feature_list:
    #     if(feature == None):
    #         continue
    #     if(len(feature) == 0):
    #         continue
    #     positive.append(feature[0])
    #     query.append(feature[1])
    #     negative.append(feature[2])
    # if(len(positive) == 0):
    #     return loss
    # positive = torch.cat(positive, dim=0)
    # positive = positive.mean(dim=0, keepdim=True)
    # query = torch.cat(query, dim=0)
    # negative = torch.cat(negative, dim=0)
    # query_length = query.shape[0]
    # with torch.no_grad():
    #     all_feat = torch.cat((positive.unsqueeze(0).repeat(query_length, 1, 1), negative.unsqueeze(0).repeat(query_length, 1, 1)), dim=1)
    # seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat, dim=2)
    # loss = loss + F.cross_entropy(seg_logits / temperature, torch.zeros(query_length).long().to(query.device))
    loss = torch.tensor(0.0)
    positive, query, negative = [], [], []
    for feature in feature_list:
        if feature == None:
            continue
        if len(feature) == 0:
            continue
        positive = feature[0]
        query = feature[1]
        negative = feature[2]
        positive = positive.mean(dim=0, keepdim=True)
        query_length = query.shape[0]
        with torch.no_grad():
            all_feat = torch.cat(
                (positive.unsqueeze(0).repeat(query_length, 1, 1), negative.unsqueeze(0).repeat(query_length, 1, 1)),
                dim=1,
            )
        seg_logits = torch.cosine_similarity(query.unsqueeze(1), all_feat, dim=2)
        loss = loss + F.cross_entropy(seg_logits / temperature, torch.zeros(query_length).long().to(query.device))
    return loss / 2

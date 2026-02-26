import torch
import torch.nn as nn
import random
import numpy as np
EPS = 1e-7
__all__ = [
    'generate_params'
]


def generate_params(model, key):
    """define a generator
    will be convert to a param list by list(params) in optimizer.add_param_group() method
    """
    if key == 'conv_weight':
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                yield m.weight
    if key == 'conv_bias':
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.bias is not None:
                yield m.bias
    if key == 'bn_weight':
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                yield m.weight
    if key == 'bn_bias':
        for n, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and m.bias is not None:
                yield m.bias
    # any type you want to specify

def mixup_data(x, y, alpha=0.4):
    '''returns mixed inputs, pairs of targets, and lambda'''
    if alpha <= 0: return x, y, None, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b, lam)

def cutmix_data(images, masks, alpha=1.0):
    """对图像和掩码做 CutMix，返回混合后的 images, masks 和 lambda"""
    if alpha <= 0:
        return images, masks, 1.0
    batch_size, C, H, W = images.size()
    # 随机打乱
    perm = torch.randperm(batch_size)
    shuffled_images = images[perm]
    shuffled_masks  = masks[perm]

    # 样本间 Beta 分布采样
    lam = np.random.beta(alpha, alpha)
    # 计算裁剪区域大小
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 计算边界
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 正确保留 channel 维
    images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
    masks[:, :, bby1:bby2, bbx1:bbx2] = shuffled_masks[:, :, bby1:bby2, bbx1:bbx2]

    # 更新 lambda 为保留原图像的面积比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return images, masks, lam, perm
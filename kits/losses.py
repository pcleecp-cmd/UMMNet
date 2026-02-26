# ──────────  kits/losses.py  ──────────
import torch, torch.nn as nn, torch.nn.functional as F
from .utils import EPS                # utils.py 里放：EPS = 1e-7

__all__ = [
    "DiceLoss", "BCELoss", "FocalBCELoss",
    "TverskyLoss", "BoundaryLoss",
]

# ---------- Dice ----------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth, self.reduction = smooth, reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
        dims = (1, 2, 3) if prob.ndim == 4 else tuple(range(1, prob.ndim))
        inter  = (prob * targets).sum(dims)
        union  = prob.sum(dims) + targets.sum(dims)
        dice   = (2 * inter + self.smooth) / (union + self.smooth + EPS)
        loss   = 1 - dice
        return loss.mean() if self.reduction == "mean" else loss.sum()

# ---------- “普通” BCE ----------
class BCELoss(nn.Module):
    def __init__(self, pos_weight=None, reduction: str = "mean"):
        super().__init__()
        self.fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    def forward(self, logits, targets):
        return self.fn(logits, targets)

# ---------- Focal BCE ----------
class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma, self.reduction = gamma, reduction

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob  = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
        p_t   = prob * targets + (1 - prob) * (1 - targets)
        focal = (1 - p_t) ** self.gamma
        loss  = focal * bce
        return (loss.mean() if self.reduction == "mean"
                else loss.sum() if self.reduction == "sum" else loss)

# ---------- Tversky ----------
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, reduction="mean"):
        super().__init__()
        self.alpha, self.beta, self.smooth, self.reduction = alpha, beta, smooth, reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
        dims = (1, 2, 3) if prob.ndim == 4 else tuple(range(1, prob.ndim))
        TP = (prob * targets).sum(dims)
        FP = ((1 - targets) * prob).sum(dims)
        FN = (targets * (1 - prob)).sum(dims)
        score = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth + EPS)
        loss  = 1 - score
        return loss.mean() if self.reduction == "mean" else loss.sum()

# ---------- Boundary ----------
class BoundaryLoss(nn.Module):
    """L1 on (x - avgpool(x))"""
    def forward(self, logits, targets):
        logits  = logits.float()
        targets = targets.float()
        pred_b  = logits - F.avg_pool2d(logits, 3, 1, 1)
        gt_b    = targets - F.avg_pool2d(targets, 3, 1, 1)
        return F.l1_loss(pred_b, gt_b)

class WiouWbceLoss(torch.nn.Module):
    def __init__(self):
        super(WiouWbceLoss, self).__init__()

    def forward(self, input, target): # 'input' 是 logits
        # weit: 权重图，强调边缘或变化区域
        weit = 1 + 5 * torch.abs(f.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)

        # wbce: 加权 BCE (期望 logits 输入)
        wbce = f.binary_cross_entropy_with_logits(input, target, reduce='none') # reduce='none' 保留像素级损失
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) # 按 weit 加权平均

        # wiou: 加权 IoU
        input_probs = torch.sigmoid(input) # IoU 需要概率
        inter = ((input_probs * target) * weit).sum(dim=(2, 3))
        union = ((input_probs + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-7) # 加入平滑项防止除零
        return (wbce + wiou).mean() # 对批次中的样本取平均
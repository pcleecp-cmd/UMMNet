# UMMNet-main/UMMNet/kits/__init__.py
# 恢复到只配置 DiceLoss 和 BCELoss 的状态

# --- 从同级目录的其他模块导入 ---
from .metrics import *
from .schedulers import LR_Scheduler # 保留导入，以防其他地方用到
# --- 只导入实际定义的损失类 ---
from .losses import TverskyLoss # 正确的二分类交叉熵类名
from .losses import DiceLoss # 正确的 Dice 类名
from .losses import FocalBCELoss,BCELoss,BoundaryLoss,WiouWbceLoss
# --- 其他工具类导入 ---
from .utils import generate_params
from .logger import setup_logger
from .saver import Saver
from .summaries import TensorboardSummary

# 定义可识别的损失名称 (只保留 bce 和 dice)
loss_names = ('bce', 'dice', 'focal', 'dice+focal')


def configure_loss(loss_name, weight=None, **kwargs):
    from .losses import FocalBCELoss
    _loss = None
    loss_name = loss_name.lower()
    print(f"Configuring loss (from kits/__init__): {loss_name}")

    if loss_name == 'bce':
        # 使用 BCELoss
        _loss = BCELoss(pos_weight=torch.tensor(weight) if weight is not None else None)
    elif loss_name == 'focal':
        from .losses import FocalBCELoss
        _loss = FocalBCELoss(gamma=2.0)

    elif loss_name == 'dice+focal':

        dice = DiceLoss(**kwargs)
        focal = FocalBCELoss(gamma=2.0)
        _loss = lambda o, y: 0.5 * dice(o, y) + 0.5 * focal(o, y)
    else:
        raise NotImplementedError('loss {} is not recognized. Available: {}'.format(loss_name, loss_names))

    return _loss

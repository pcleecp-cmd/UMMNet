# train_code.py
from medpy.metric.binary import jc as iou, hd95 as hausdorff95
import numpy as np
import os
import random
import copy
import json
import csv
import traceback
import gc
import math
import time
import inspect
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import GradScaler, autocast

from kits.losses import DiceLoss, BCELoss, WiouWbceLoss
from kits import SegMetrics, Saver, TensorboardSummary
from data.dataset import PASD_Dataset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


torch.backends.cudnn.benchmark = True


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.slow_weights = [
            [p.clone().detach() for p in group['params']]
            for group in optimizer.param_groups
        ]
        for w_group in self.slow_weights:
            for w in w_group:
                w.requires_grad_(False)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            for (group, slow_group) in zip(self.optimizer.param_groups, self.slow_weights):
                for p, q in zip(group['params'], slow_group):
                    if p.grad is None:
                        continue
                    q.data.add_(p.data - q.data, alpha=self.alpha)
                    p.data.copy_(q.data)
        return loss

    def state_dict(self):
        return {
            'base_optimizer': self.optimizer.state_dict(),
            'slow_weights': [[w.clone() for w in wg] for wg in self.slow_weights],
            'step_counter': self.step_counter,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['base_optimizer'])
        self.slow_weights = [
            [w.clone().detach() for w in wg]
            for wg in state_dict['slow_weights']
        ]
        self.step_counter = state_dict['step_counter']


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
    except AttributeError:
        pass
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def patient_id_from_sample_id(sample_id: str) -> str:
    m = re.match(r"(sub\d+|v\d+)", sample_id)
    return m.group(1) if m else sample_id.split("_")[0]


class Trainer:
    def __init__(self, args, model):
        # 1. 固定随机种子
        set_seed(args.seed)
        self.args = args
        self.start_epoch = 0
        self.current_epoch = 0

        # 2. 训练总轮数 & 动态数据增强起始轮
        self.total_epochs = args.epochs
        self.advanced_aug_start_epoch_for_ds = int(self.total_epochs * 0.2)

        # 3. 加载超参配置
        self._load_config(args)

        # 4. 设备 & 模型
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        forward_params = inspect.signature(self.model.forward).parameters
        self.model_accepts_current_epoch = "current_epoch" in forward_params
        self.model_accepts_save_analysis = "save_analysis" in forward_params
        # Keep initialization logging minimal.

        # 5. 数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders(args)
        # Keep initialization logging minimal.

        # 6. Saver & TensorBoard
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # 7. 优化器、调度器
        self._build_optimizer_and_scheduler(args)

        # 8. SWA
        self.swa_start = int(0.5 * self.total_epochs)
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=args.lr * 0.1)

        # 9. 损失 & AMP
        self._build_losses(args)
        self.use_amp = args.amp
        # Keep initialization logging minimal.
        self.scaler = GradScaler(
            enabled=self.use_amp,
            init_scale=2 ** 13,
            growth_interval=1000,
            backoff_factor=0.5
        )

        # 10. 评估指标 & EMA
        self.evaluator = SegMetrics(num_class=args.num_classes + 1)
        self.ema_decay = 0.99
        self.ema = self._build_ema_model(model)

        # 11. checkpoint 恢复
        self.best_dice = 0.0
        self.best_pred = 0.0
        if args.resume:
            self._resume_checkpoint(args.resume, args.ft)

        # 12. 预置动态属性
        self.aug_strength = 0.0
        self.bl_weight = 0.0
        self.last_eval_result = {}
        self.loss_history = {}

    def _build_ema_model(self, model):
        """创建EMA模型，针对MRI数据集调整衰减率"""
        # 将衰减率从0.995降低到0.99，使模型更快适应最新参数
        # 对于MRI这种细节丰富的医学图像，模型需要更快适应最新学到的特征
        self.ema_decay = 0.99  # 修改为0.99，适合医学图像数据集通常较小的情况
        
        ema = copy.deepcopy(model).eval().to(self.device)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    def _forward_model(self, x, epoch, save_analysis: bool = False):
        """Unify forward call across models with/without current_epoch arg."""
        kwargs = {}
        if self.model_accepts_current_epoch:
            kwargs["current_epoch"] = epoch
        if self.model_accepts_save_analysis:
            kwargs["save_analysis"] = bool(save_analysis)
        return self.model(x, **kwargs)

    def _update_ema_model(self):
        """更新EMA模型权重，使用动态衰减率"""
        # 根据训练进度动态调整衰减率，训练初期使用较低的衰减率以快速学习
        epoch_progress = min(self.current_epoch / (self.total_epochs * 0.6), 1.0)
        current_decay = 0.99 + 0.01 * epoch_progress  # 从0.99逐渐增加到1.0
        current_decay = min(current_decay, 0.9995)    # 设置上限
        
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
                ema_p.data.mul_(current_decay).add_(p.data, alpha=1 - current_decay)

    @staticmethod
    def _snapshot_bn_running_stats(model):
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if (
                    module.running_mean is None
                    or module.running_var is None
                    or module.num_batches_tracked is None
                ):
                    continue
                stats[name] = {
                    "running_mean": module.running_mean.detach().clone(),
                    "running_var": module.running_var.detach().clone(),
                    "num_batches_tracked": int(module.num_batches_tracked.detach().item()),
                }
        return stats

    @staticmethod
    def _restore_bn_running_stats(model, snapshot):
        if not snapshot:
            return
        with torch.no_grad():
            for name, module in model.named_modules():
                if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                    continue
                if name not in snapshot:
                    continue
                if (
                    module.running_mean is None
                    or module.running_var is None
                    or module.num_batches_tracked is None
                ):
                    continue
                state = snapshot[name]
                module.running_mean.copy_(state["running_mean"])
                module.running_var.copy_(state["running_var"])
                module.num_batches_tracked.fill_(state["num_batches_tracked"])

    @staticmethod
    def _seed_worker(worker_id, base_seed):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _load_config(self, args):
        # 边缘权重、平滑项、梯度裁剪等
        self.edge_kernel_size = getattr(args, 'edge_kernel_size', 9)
        self.edge_multiplier = getattr(args, 'edge_multiplier', 2.5)
        # 统一参数命名
        self.edge_weight_clamp = (
            getattr(args, 'edge_weight_min', 1.0), 
            getattr(args, 'edge_weight_max', 4.0)
        )
        self.dice_smooth = getattr(args, 'dice_smooth', 100.0)
        self.dice_eps = getattr(args, 'dice_eps', 1e-7)
        self.grad_clip_norm = getattr(args, 'grad_clip_norm', 2.0)
        self.enable_batch_safety_checks = bool(getattr(args, 'enable_batch_safety_checks', False))
        self.snapshot_bn_for_recovery = bool(getattr(args, 'snapshot_bn_for_recovery', False))

        # Keep config loading silent for clean epoch-level logging.

    def _build_dataloaders(self, args):
        split_seed = int(getattr(args, "split_seed", 42))
        self.split_seed = split_seed
        self.args.split_seed = split_seed
        # Keep split config logging minimal.

        base_kwargs = dict(
            data_root=args.data_root,
            img_size=(args.img_size, args.img_size),
            t2_img_dir_rel=args.t2_img_dir_rel,
            bssfp_img_dir_rel=args.reg_bssfp_img_dir_rel,
            t2_mask_dir_rel=args.t2_mask_dir_rel,
            bssfp_mask_dir_rel=args.reg_bssfp_mask_dir_rel,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=split_seed,
        )

        ds_kwargs = lambda split, aug: dict(
            split=split, use_augmentation=aug,
            advanced_aug_epoch_threshold=self.advanced_aug_start_epoch_for_ds,
            **base_kwargs
        )

        train_ds = PASD_Dataset(**ds_kwargs('train', True))
        val_ds = PASD_Dataset(**ds_kwargs('val', False))
        test_ds = PASD_Dataset(**ds_kwargs('test', False))

        worker_fn = lambda wid: Trainer._seed_worker(wid, args.seed)
        def make_loader(ds, shuffle, drop_last):
            loader_kwargs = dict(
                dataset=ds,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=drop_last,
                worker_init_fn=worker_fn,
            )
            if args.num_workers > 0:
                loader_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
                loader_kwargs["prefetch_factor"] = max(2, int(getattr(args, "prefetch_factor", 4)))
            return data.DataLoader(**loader_kwargs)

        return (
            make_loader(train_ds, shuffle=True,  drop_last=True),
            make_loader(val_ds,   shuffle=False, drop_last=False),
            make_loader(test_ds,  shuffle=False, drop_last=False),
        )

    def _build_optimizer_and_scheduler(self, args):
        base_opt = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.base_optimizer = base_opt
        self.optimizer = Lookahead(base_opt, k=5, alpha=0.5)

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.total_epochs
        multiplier = 4
        warmup_ep = 10
        plateau_pct = 0.30

        warmup_steps = warmup_ep * steps_per_epoch
        plateau_steps = int(total_steps * plateau_pct)
        min_mul = 0.3

        def two_stage_lr(step: int):
            if step < warmup_steps:
                return 1 + (multiplier - 1) * step / warmup_steps
            elif step < warmup_steps + plateau_steps:
                return float(multiplier)
            else:
                prog = (step - warmup_steps - plateau_steps) / (total_steps - warmup_steps - plateau_steps)
                return (min_mul * multiplier +
                        (multiplier - min_mul * multiplier) * 0.5 * (1 + math.cos(math.pi * prog)))

        self.scheduler = LambdaLR(base_opt, lr_lambda=two_stage_lr)

        # Scheduler setup is intentionally silent.

    def _build_losses(self, args):
        # Dice
        self.dice_fn = DiceLoss()

        # BCE with optional pos_weight
        if hasattr(args, 'bce_pos_weight_value') and args.bce_pos_weight_value > 0:
            self.effective_bce_pos_weight = torch.tensor([args.bce_pos_weight_value], device=self.device)
            self.bce_fn = BCELoss(pos_weight=self.effective_bce_pos_weight)
        else:
            self.effective_bce_pos_weight = None
            self.bce_fn = BCELoss()

        # For fallback
        self.bce_fn_for_fallback = self.bce_fn

    def compute_weighted_loss(self, logits, targets):
        """统一计算加权 BCE+Dice，包含 fallback"""
        if (not torch.isfinite(logits).all()) or (not torch.isfinite(targets).all()):
            return torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)
        weights = self.compute_adaptive_edge_weights(targets)
        loss_bce = self.compute_stable_bce_loss(logits, targets, weights)
        loss_dice = self.compute_stable_dice_loss(logits, targets, weights)
        main_loss = self.dice_weight * loss_dice + self.bce_weight * loss_bce

        if not torch.isfinite(main_loss):
            # fallback：纯稳定版本
            loss_dice_fb = self.compute_stable_dice_loss(logits, targets)
            loss_bce_fb = self.compute_stable_bce_loss(logits, targets)
            main_loss = self.dice_weight * loss_dice_fb + self.bce_weight * loss_bce_fb
            if not torch.isfinite(main_loss):
                return torch.tensor(float("nan"), device=logits.device, dtype=logits.dtype)

        return main_loss

    def _epoch_setup(self, epoch):
        """每个 epoch 开始前的动态设置"""

        # Loss weights
        self.dice_weight = getattr(self.args, 'dice_weight', 0.75)
        self.bce_weight = 1.0 - self.dice_weight

        # Keep epoch setup silent.

    def compute_adaptive_edge_weights(self, targets):
        with torch.no_grad():
            B, C, H, W = targets.shape
            pad = self.edge_kernel_size // 2
            pooled = F.avg_pool2d(targets, kernel_size=self.edge_kernel_size,
                                  stride=1, padding=pad)
            edge_strength = torch.abs(pooled - targets)

            t2 = targets * targets
            pooled2 = F.avg_pool2d(t2, kernel_size=self.edge_kernel_size,
                                   stride=1, padding=pad)
            local_var = pooled2 - pooled * pooled
            local_var = torch.clamp(local_var, min=1e-6)

            adaptive = self.edge_multiplier * (1.0 + 0.5 / (1.0 + 10.0 * local_var))
            raw_w = 1.0 + adaptive * edge_strength
            smooth_w = F.avg_pool2d(raw_w, kernel_size=3, stride=1, padding=1)
            return torch.clamp(smooth_w, *self.edge_weight_clamp)

    def compute_stable_dice_loss(self, logits, targets, weights=None):
        probs = torch.sigmoid(logits).clamp(self.dice_eps, 1.0 - self.dice_eps)
        targets = targets.clamp(self.dice_eps, 1.0 - self.dice_eps)

        if weights is not None:
            wsum = weights.sum(dim=(1,2,3), keepdim=True).clamp(min=1.0)
            H, W = targets.shape[2:]
            normalized = weights * (H * W) / wsum
            inter = (probs * targets * normalized).sum(dim=(1,2,3))
            p_sum = (probs * normalized).sum(dim=(1,2,3))
            t_sum = (targets * normalized).sum(dim=(1,2,3))
        else:
            inter = (probs * targets).sum(dim=(1,2,3))
            p_sum = probs.sum(dim=(1,2,3))
            t_sum = targets.sum(dim=(1,2,3))

        denom = (p_sum + t_sum + self.dice_smooth).clamp(min=self.dice_smooth)
        dice_score = (2.0 * inter + self.dice_smooth) / denom
        return 1.0 - dice_score.mean()

    def compute_stable_bce_loss(self, logits, targets, weights=None):
        if self.effective_bce_pos_weight is not None:
            pixel_bce = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.effective_bce_pos_weight,
                reduction='none'
            )
        else:
            pixel_bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )

        if weights is not None:
            wb = pixel_bce * weights
            wsum = weights.sum(dim=(1,2,3), keepdim=True).clamp(min=1.0)
            return (wb.sum(dim=(1,2,3)) / wsum.squeeze()).mean()
        else:
            return pixel_bce.mean()

    def _phase_to_tag(self, phase: str) -> str:
        if phase.startswith("测试-EMA"):
            return "test_ema"
        if phase.startswith("测试"):
            return "test"
        if phase.startswith("验证"):
            return "val"
        return phase.lower()

    def _experiment_file(self, filename: str) -> str:
        return os.path.join(self.saver.experiment_dir, filename)

    @staticmethod
    def _write_csv(path: str, rows, fieldnames=None):
        if not rows:
            return
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _write_json(path: str, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def update_loss_history(
        self,
        epoch: int,
        train_loss: float = None,
        val_loss: float = None,
        val_dice: float = None,
        is_best: bool = False,
    ):
        key = int(epoch)
        prev = self.loss_history.get(
            key,
            {
                "epoch": key,
                "train_loss": float("nan"),
                "val_loss": float("nan"),
                "val_dice": float("nan"),
                "is_best": 0,
            },
        )
        row = dict(prev)
        if train_loss is not None:
            row["train_loss"] = float(train_loss)
        if val_loss is not None:
            row["val_loss"] = float(val_loss)
        if val_dice is not None:
            row["val_dice"] = float(val_dice)
        if is_best:
            row["is_best"] = 1
        self.loss_history[key] = row

        rows = [self.loss_history[k] for k in sorted(self.loss_history.keys())]
        self._write_csv(
            self._experiment_file("loss_history.csv"),
            rows,
            fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "is_best"],
        )

    @staticmethod
    def _compute_case_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
        pred = np.asarray(pred_mask).astype(bool)
        gt = np.asarray(gt_mask).astype(bool)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]

        tp = int(np.logical_and(pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou_score = tp / (tp + fp + fn + 1e-8)

        try:
            if pred.any() and gt.any():
                hd = float(hausdorff95(pred, gt))
            elif not pred.any() and not gt.any():
                hd = 0.0
            else:
                hd = float(np.linalg.norm(pred.shape))
        except Exception:
            hd = float("nan")

        return {
            "dice": float(dice),
            "iou": float(iou_score),
            "hd95": hd,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    @staticmethod
    def _compute_boundary_band(mask: torch.Tensor, band_width: int) -> torch.Tensor:
        # mask: [B,1,H,W], 0/1
        band_width = max(1, int(band_width))
        kernel = 2 * band_width + 1
        m = (mask > 0.5).float()
        dilated = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=band_width)
        eroded = 1.0 - F.max_pool2d(1.0 - m, kernel_size=kernel, stride=1, padding=band_width)
        return (dilated - eroded) > 0.5

    def _get_model_analysis_cache(self):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, "get_analysis_cache"):
            return model_ref.get_analysis_cache()
        return {}

    @staticmethod
    def _build_binned_curve(u_values, e_values, num_bins: int, bin_edges=None):
        if len(u_values) == 0:
            return [], [], None
        u = np.asarray(u_values, dtype=np.float64)
        e = np.asarray(e_values, dtype=np.float64)
        valid = np.isfinite(u) & np.isfinite(e)
        u = u[valid]
        e = e[valid]
        if u.size == 0:
            return [], [], None
        if bin_edges is None:
            num_bins = max(2, int(num_bins))
            quantiles = np.linspace(0.0, 1.0, num_bins + 1)
            edges = np.quantile(u, quantiles)
            # Avoid zero-width bins when many identical values exist.
            edges[0] = edges[0] - 1e-8
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-8
        else:
            edges = np.asarray(bin_edges, dtype=np.float64)
            num_bins = len(edges) - 1

        rows = []
        per_bin_means = []
        for b in range(num_bins):
            lo = edges[b]
            hi = edges[b + 1]
            if b == num_bins - 1:
                mask = (u >= lo) & (u <= hi)
            else:
                mask = (u >= lo) & (u < hi)
            if mask.any():
                mu = float(u[mask].mean())
                me = float(e[mask].mean())
                cnt = int(mask.sum())
            else:
                mu = float("nan")
                me = float("nan")
                cnt = 0
            rows.append(
                {
                    "bin": b,
                    "u_low": float(lo),
                    "u_high": float(hi),
                    "u_mean": mu,
                    "error_mean": me,
                    "count": cnt,
                }
            )
            per_bin_means.append(me)
        return rows, per_bin_means, edges

    def _maybe_plot_curve(self, rows, out_png: str, title: str, y_label: str, std_values=None):
        if plt is None or not rows:
            return False
        xs = [r["bin"] for r in rows]
        ys = [r["error_mean"] for r in rows]
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, marker="o", linewidth=2)
        if std_values is not None and len(std_values) == len(ys):
            ys_np = np.asarray(ys, dtype=np.float64)
            std_np = np.asarray(std_values, dtype=np.float64)
            ax.fill_between(xs, ys_np - std_np, ys_np + std_np, alpha=0.2)
        ax.set_xlabel("Uncertainty Bin (low -> high)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        return True

    def training(self, epoch):
        self.current_epoch = epoch
        self._epoch_setup(epoch)
        self.model.train()
        total_loss = 0.0
        valid_steps = 0
        for i, (img, tgt, _) in enumerate(self.train_loader):
            img = img.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)

            # NaN/Inf 检查
            if self.enable_batch_safety_checks and ((not torch.isfinite(img).all()) or (not torch.isfinite(tgt).all())):
                print(f"WARNING: non-finite input at step {i}, skipped.")
                continue

            img_mixed, tgt_for_loss = img, tgt

            self.optimizer.zero_grad(set_to_none=True)
            bn_snapshot = self._snapshot_bn_running_stats(self.model) if self.snapshot_bn_for_recovery else None
            with autocast(enabled=self.use_amp):
                logits = self._forward_model(img_mixed, epoch)
            if self.enable_batch_safety_checks and (not torch.isfinite(logits).all()):
                print(f"WARNING: non-finite logits at step {i}, skipped.")
                if bn_snapshot is not None:
                    self._restore_bn_running_stats(self.model, bn_snapshot)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue
            with autocast(enabled=False):
                loss = self.compute_weighted_loss(logits.float(), tgt_for_loss.float())
            if self.enable_batch_safety_checks and (not torch.isfinite(loss)):
                print(f"WARNING: non-finite loss at step {i}, skipped.")
                if bn_snapshot is not None:
                    self._restore_bn_running_stats(self.model, bn_snapshot)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            # 反向 + 梯度裁剪
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                      max_norm=self.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                print(f"WARNING: non-finite grad norm at step {i}, skipped.")
                if bn_snapshot is not None:
                    self._restore_bn_running_stats(self.model, bn_snapshot)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # SWA update
            if epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            # EMA update - 使用优化后的方法
            self._update_ema_model()

            loss_val = loss.item()
            total_loss += loss_val
            valid_steps += 1

        avg_loss = total_loss / max(1, valid_steps)
        if self.writer:
            self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        return avg_loss

    def _evaluate(self, epoch, loader, phase="验证"):
        self.model.eval()
        self.evaluator.reset()
        total_loss = 0.0
        valid_eval_batches = 0
        all_iou, all_hd95 = [], []
        total_TP = total_TN = total_FP = total_FN = 0
        measure_time = False
        save_eval_artifacts = bool(getattr(self.args, "save_eval_artifacts", False))
        need_uncertainty_curve = False
        need_gamma_stats = False
        need_analysis = need_uncertainty_curve or need_gamma_stats
        band_width = int(getattr(self.args, "uncertainty_band_width", 4))
        uncertainty_bins = int(getattr(self.args, "uncertainty_bins", 10))
        total_forward_time = 0.0
        total_forward_samples = 0
        phase_tag = self._phase_to_tag(phase)

        slice_rows = []
        patient_to_case_metrics = defaultdict(list)
        patient_uncertainty = defaultdict(lambda: {"u": [], "e": []})
        gamma_collector = defaultdict(
            lambda: {
                "labels": [],
                "boundary": {
                    "dir_sum": defaultdict(float),
                    "dir_sq_sum": defaultdict(float),
                    "dir_count": defaultdict(int),
                    "entropy_sum": 0.0,
                    "entropy_sq_sum": 0.0,
                    "entropy_count": 0,
                    "entropy_samples": [],
                },
                "interior": {
                    "dir_sum": defaultdict(float),
                    "dir_sq_sum": defaultdict(float),
                    "dir_count": defaultdict(int),
                    "entropy_sum": 0.0,
                    "entropy_sq_sum": 0.0,
                    "entropy_count": 0,
                    "entropy_samples": [],
                },
            }
        )
        gamma_label_mismatch_warned = set()
        entropy_sample_cap = int(getattr(self.args, "gamma_entropy_sample_cap", 2000))

        with torch.no_grad():
            for i, (img, tgt, metas) in enumerate(loader):
                img = img.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                if self.enable_batch_safety_checks and ((not torch.isfinite(img).all()) or (not torch.isfinite(tgt).all())):
                    print(f"WARNING: {phase} non-finite input at step {i}, skipped.")
                    continue
                ids = None
                if isinstance(metas, dict) and "id" in metas:
                    ids = metas["id"]
                if ids is None:
                    ids = [f"{phase_tag}_{i}_{k}" for k in range(img.shape[0])]
                if isinstance(ids, str):
                    ids = [ids]

                if measure_time and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start_t = time.perf_counter()

                use_tta = False
                with autocast(enabled=self.use_amp):
                    logits = (self.tta_predict(img, epoch) if use_tta
                              else self._forward_model(img, epoch, save_analysis=need_analysis))
                if self.enable_batch_safety_checks and (not torch.isfinite(logits).all()):
                    print(f"WARNING: {phase} non-finite logits at step {i}, skipped.")
                    continue

                if measure_time:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    total_forward_time += (time.perf_counter() - start_t)
                    total_forward_samples += int(img.shape[0])

                loss = self.dice_fn(logits.float(), tgt.float())
                if self.enable_batch_safety_checks and (not torch.isfinite(loss)):
                    print(f"WARNING: {phase} non-finite loss at step {i}, skipped.")
                    continue
                total_loss += loss.item()
                valid_eval_batches += 1

                pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)
                gt = tgt.cpu().numpy().astype(bool)
                pred_tensor = (torch.sigmoid(logits) > 0.5)
                gt_tensor = (tgt > 0.5)

                for b, (p, g) in enumerate(zip(pred, gt)):
                    case = self._compute_case_metrics(p, g)
                    all_iou.append(case["iou"])
                    all_hd95.append(case["hd95"])
                    total_TP += case["tp"]
                    total_TN += case["tn"]
                    total_FP += case["fp"]
                    total_FN += case["fn"]

                    sample_id = str(ids[b]) if b < len(ids) else f"{phase_tag}_{i}_{b}"
                    patient_id = patient_id_from_sample_id(sample_id)
                    patient_to_case_metrics[patient_id].append(case)
                    slice_rows.append(
                        {
                            "phase": phase_tag,
                            "epoch": int(epoch),
                            "sample_id": sample_id,
                            "patient_id": patient_id,
                            "dice": case["dice"],
                            "iou": case["iou"],
                            "hd95": case["hd95"],
                            "tp": case["tp"],
                            "tn": case["tn"],
                            "fp": case["fp"],
                            "fn": case["fn"],
                        }
                    )

                self.evaluator.update(pred.astype(np.uint8), gt.astype(np.uint8))

                if need_analysis:
                    analysis_cache = self._get_model_analysis_cache()

                    if need_uncertainty_curve:
                        skip1 = (
                            analysis_cache.get("ummf", {}).get("skip1")
                            if isinstance(analysis_cache, dict)
                            else None
                        )
                        uncertainty_map = (
                            skip1.get("uncertainty_map")
                            if isinstance(skip1, dict)
                            else None
                        )
                        if uncertainty_map is not None:
                            uncertainty_map = F.interpolate(
                                uncertainty_map,
                                size=tgt.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            boundary_mask = self._compute_boundary_band(tgt, band_width)
                            error_map = (pred_tensor != gt_tensor).float()
                            for b in range(img.shape[0]):
                                sample_id = str(ids[b]) if b < len(ids) else f"{phase_tag}_{i}_{b}"
                                patient_id = patient_id_from_sample_id(sample_id)
                                band = boundary_mask[b, 0]
                                if band.any():
                                    u_vals = uncertainty_map[b, 0][band].detach().cpu().numpy()
                                    e_vals = error_map[b, 0][band].detach().cpu().numpy()
                                    patient_uncertainty[patient_id]["u"].extend(
                                        u_vals.astype(np.float64).tolist()
                                    )
                                    patient_uncertainty[patient_id]["e"].extend(
                                        e_vals.astype(np.float64).tolist()
                                    )

                    if need_gamma_stats:
                        udec_dict = (
                            analysis_cache.get("udec", {})
                            if isinstance(analysis_cache, dict)
                            else {}
                        )
                        boundary_mask_full = self._compute_boundary_band(tgt, band_width)
                        interior_mask_full = gt_tensor & (~boundary_mask_full)

                        for stage_name, stage_payload in udec_dict.items():
                            if not isinstance(stage_payload, dict):
                                continue
                            gamma = stage_payload.get("gamma")
                            labels = stage_payload.get("direction_labels", [])
                            if not isinstance(labels, (list, tuple)):
                                labels = []
                            else:
                                labels = list(labels)
                            if gamma is None:
                                continue
                            if gamma.ndim != 4:
                                key = (stage_name, "ndim", int(getattr(gamma, "ndim", -1)))
                                if key not in gamma_label_mismatch_warned:
                                    print(
                                        f"WARNING: gamma_stats: stage={stage_name} gamma.ndim={getattr(gamma, 'ndim', -1)}，"
                                        "预期4D，跳过该stage。"
                                    )
                                    gamma_label_mismatch_warned.add(key)
                                continue
                            d_channels = int(gamma.shape[1])
                            if d_channels <= 0:
                                continue
                            if len(labels) != d_channels:
                                key = (stage_name, len(labels), d_channels)
                                if key not in gamma_label_mismatch_warned:
                                    print(
                                        f"WARNING: gamma_stats: stage={stage_name} direction_labels={len(labels)} "
                                        f"与gamma通道数={d_channels}不一致，已自动对齐。"
                                    )
                                    gamma_label_mismatch_warned.add(key)
                                if len(labels) < d_channels:
                                    labels.extend(
                                        [f"dir_{idx}" for idx in range(len(labels), d_channels)]
                                    )
                                else:
                                    labels = labels[:d_channels]
                            if not gamma_collector[stage_name]["labels"]:
                                gamma_collector[stage_name]["labels"] = list(labels)
                            g_h, g_w = gamma.shape[-2:]
                            boundary_mask = (
                                F.interpolate(
                                    boundary_mask_full.float(),
                                    size=(g_h, g_w),
                                    mode="nearest",
                                )
                                > 0.5
                            )
                            interior_mask = (
                                F.interpolate(
                                    interior_mask_full.float(),
                                    size=(g_h, g_w),
                                    mode="nearest",
                                )
                                > 0.5
                            )

                            gamma_np = gamma.detach().cpu().numpy()  # [B, D, H, W]
                            for b in range(gamma_np.shape[0]):
                                for region_name, region_mask in (
                                    ("boundary", boundary_mask[b, 0]),
                                    ("interior", interior_mask[b, 0]),
                                ):
                                    region_mask_np = region_mask.detach().cpu().numpy().astype(bool)
                                    if not region_mask_np.any():
                                        continue
                                    # Use flattened masking to keep direction axis stable as [D, N].
                                    gamma_flat = gamma_np[b].reshape(d_channels, -1)  # [D, H*W]
                                    mask_flat = region_mask_np.reshape(-1)
                                    if gamma_flat.shape[1] != mask_flat.size:
                                        key = (stage_name, "mask_size", gamma_flat.shape[1], mask_flat.size)
                                        if key not in gamma_label_mismatch_warned:
                                            print(
                                                f"WARNING: gamma_stats: stage={stage_name} mask大小与特征不一致 "
                                                f"({mask_flat.size} vs {gamma_flat.shape[1]})，跳过该region。"
                                            )
                                            gamma_label_mismatch_warned.add(key)
                                        continue
                                    pix = gamma_flat[:, mask_flat]  # [D, N]
                                    if pix.size == 0:
                                        continue
                                    labels_cur = list(labels)
                                    entropy = -(pix * np.log(pix + 1e-6)).sum(axis=0)
                                    region_stats = gamma_collector[stage_name][region_name]
                                    entropy = entropy.astype(np.float64)
                                    region_stats["entropy_sum"] += float(entropy.sum())
                                    region_stats["entropy_sq_sum"] += float((entropy ** 2).sum())
                                    region_stats["entropy_count"] += int(entropy.size)
                                    keep = entropy_sample_cap - len(region_stats["entropy_samples"])
                                    if keep > 0 and entropy.size > 0:
                                        if entropy.size <= keep:
                                            sampled = entropy
                                        else:
                                            idxs = np.random.choice(entropy.size, size=keep, replace=False)
                                            sampled = entropy[idxs]
                                        region_stats["entropy_samples"].extend(sampled.tolist())
                                    for d_idx, d_name in enumerate(labels_cur):
                                        vals = pix[d_idx].astype(np.float64)
                                        region_stats["dir_sum"][d_name] += float(vals.sum())
                                        region_stats["dir_sq_sum"][d_name] += float((vals ** 2).sum())
                                        region_stats["dir_count"][d_name] += int(vals.size)

        if valid_eval_batches == 0:
            print(f"WARNING: {phase} all batches skipped due to non-finite values.")
            self.last_eval_result = {
                "phase": phase_tag,
                "epoch": int(epoch),
                "error": "all_batches_skipped_non_finite",
            }
            return 0.0

        dice = self.evaluator.dice()
        precision = total_TP / (total_TP + total_FP + 1e-8)
        sensitivity = total_TP / (total_TP + total_FN + 1e-8)
        specificity = total_TN / (total_TN + total_FP + 1e-8)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
        avg_loss = total_loss / valid_eval_batches
        mean_iou = float(np.nanmean(all_iou))
        mean_hd95 = float(np.nanmean(all_hd95))

        patient_rows = []
        for patient_id, cases in sorted(patient_to_case_metrics.items()):
            dice_vals = [c["dice"] for c in cases]
            iou_vals = [c["iou"] for c in cases]
            hd_vals = [c["hd95"] for c in cases]
            patient_rows.append(
                {
                    "phase": phase_tag,
                    "epoch": int(epoch),
                    "patient_id": patient_id,
                    "num_slices": len(cases),
                    "dice_mean": float(np.nanmean(dice_vals)),
                    "dice_std": float(np.nanstd(dice_vals)),
                    "iou_mean": float(np.nanmean(iou_vals)),
                    "iou_std": float(np.nanstd(iou_vals)),
                    "hd95_mean": float(np.nanmean(hd_vals)),
                    "hd95_std": float(np.nanstd(hd_vals)),
                }
            )

        patient_dice_means = [r["dice_mean"] for r in patient_rows]
        patient_iou_means = [r["iou_mean"] for r in patient_rows]
        patient_hd_means = [r["hd95_mean"] for r in patient_rows]
        patient_summary = {
            "dice_mean_of_patient_means": float(np.nanmean(patient_dice_means)) if patient_dice_means else float("nan"),
            "dice_std_of_patient_means": float(np.nanstd(patient_dice_means)) if patient_dice_means else float("nan"),
            "iou_mean_of_patient_means": float(np.nanmean(patient_iou_means)) if patient_iou_means else float("nan"),
            "iou_std_of_patient_means": float(np.nanstd(patient_iou_means)) if patient_iou_means else float("nan"),
            "hd95_mean_of_patient_means": float(np.nanmean(patient_hd_means)) if patient_hd_means else float("nan"),
            "hd95_std_of_patient_means": float(np.nanstd(patient_hd_means)) if patient_hd_means else float("nan"),
        }

        if self.writer:
            base = phase_tag
            self.writer.add_scalar(f'{base}/loss', avg_loss, epoch)
            self.writer.add_scalar(f'{base}/dice', dice, epoch)
            self.writer.add_scalar(f'{base}/iou', mean_iou, epoch)
            self.writer.add_scalar(f'{base}/hd95', mean_hd95, epoch)
            self.writer.add_scalar(f'{base}/precision', precision, epoch)
            self.writer.add_scalar(f'{base}/sensitivity', sensitivity, epoch)
            self.writer.add_scalar(f'{base}/specificity', specificity, epoch)
            self.writer.add_scalar(f'{base}/f1_score', f1_score, epoch)
            if measure_time and total_forward_samples > 0:
                ms_per_img = 1000.0 * total_forward_time / total_forward_samples
                self.writer.add_scalar(f'{base}/inference_ms_per_image', ms_per_img, epoch)

        inference_stats = {}
        if measure_time and total_forward_samples > 0:
            ms_per_img = 1000.0 * total_forward_time / total_forward_samples
            imgs_per_sec = total_forward_samples / max(total_forward_time, 1e-12)
            inference_stats = {
                "total_seconds": float(total_forward_time),
                "num_images": int(total_forward_samples),
                "ms_per_image": float(ms_per_img),
                "images_per_second": float(imgs_per_sec),
            }
        elif measure_time:
            inference_stats = {
                "total_seconds": 0.0,
                "num_images": 0,
                "ms_per_image": float("nan"),
                "images_per_second": float("nan"),
            }

        summary_payload = {
            "phase": phase_tag,
            "epoch": int(epoch),
            "slice_count": int(len(slice_rows)),
            "patient_count": int(len(patient_rows)),
            "loss": float(avg_loss),
            "dice": float(dice),
            "iou_mean_slice": float(mean_iou),
            "hd95_mean_slice": float(mean_hd95),
            "precision": float(precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1_score),
            "patient_summary": patient_summary,
            "inference": inference_stats,
        }

        is_val_phase = phase.startswith("验证")
        is_val_best_artifact = bool(is_val_phase and (dice > self.best_pred))
        if is_val_phase:
            artifact_prefix = "val_best"
            save_base_artifacts = bool(save_eval_artifacts and is_val_best_artifact)
        else:
            artifact_prefix = phase_tag
            save_base_artifacts = bool(save_eval_artifacts)

        if save_base_artifacts:
            self._write_csv(
                self._experiment_file(f"{artifact_prefix}_slice_metrics.csv"),
                slice_rows,
            )
            self._write_csv(
                self._experiment_file(f"{artifact_prefix}_patient_metrics.csv"),
                patient_rows,
            )

        if need_uncertainty_curve and patient_uncertainty:
            all_u = []
            all_e = []
            for pe in patient_uncertainty.values():
                all_u.extend(pe["u"])
                all_e.extend(pe["e"])
            curve_rows, _, global_edges = self._build_binned_curve(
                all_u, all_e, uncertainty_bins
            )

            patient_curve_rows = []
            patient_bin_means = []
            for patient_id, pe in sorted(patient_uncertainty.items()):
                rows_p, means_p, _ = self._build_binned_curve(
                    pe["u"], pe["e"], uncertainty_bins, bin_edges=global_edges
                )
                patient_bin_means.append(means_p)
                for r in rows_p:
                    patient_curve_rows.append(
                        {
                            "phase": phase_tag,
                            "epoch": int(epoch),
                            "patient_id": patient_id,
                            **r,
                        }
                    )

            bin_std = []
            if patient_bin_means:
                arr = np.asarray(patient_bin_means, dtype=np.float64)
                bin_std = np.nanstd(arr, axis=0).tolist()
                for i_bin, std_val in enumerate(bin_std):
                    if i_bin < len(curve_rows):
                        curve_rows[i_bin]["error_std_across_patients"] = float(std_val)

            curve_csv = self._experiment_file(f"{artifact_prefix}_uncertainty_error_curve.csv")
            patient_curve_csv = self._experiment_file(
                f"{artifact_prefix}_uncertainty_error_curve_per_patient.csv"
            )
            self._write_csv(curve_csv, curve_rows)
            self._write_csv(patient_curve_csv, patient_curve_rows)
            plotted = self._maybe_plot_curve(
                curve_rows,
                self._experiment_file(f"{artifact_prefix}_uncertainty_error_curve.png"),
                title="Boundary-band Uncertainty-Error Curve",
                y_label="Boundary Error Rate",
                std_values=bin_std if bin_std else None,
            )
            summary_payload["uncertainty_curve"] = {
                "rows_csv": os.path.basename(curve_csv),
                "per_patient_csv": os.path.basename(patient_curve_csv),
                "plot_saved": bool(plotted),
            }

        if need_gamma_stats and gamma_collector:
            gamma_rows = []
            entropy_box_rows = []
            for stage_name, stage_data in sorted(gamma_collector.items()):
                labels = stage_data.get("labels", [])
                for region_name in ("boundary", "interior"):
                    region_stats = stage_data[region_name]
                    row = {
                        "phase": phase_tag,
                        "epoch": int(epoch),
                        "stage": stage_name,
                        "region": region_name,
                    }
                    entropy_count = int(region_stats["entropy_count"])
                    if entropy_count > 0:
                        entropy_mean = region_stats["entropy_sum"] / entropy_count
                        entropy_var = max(
                            region_stats["entropy_sq_sum"] / entropy_count - entropy_mean ** 2,
                            0.0,
                        )
                        entropy_std = math.sqrt(entropy_var)
                    else:
                        entropy_mean = float("nan")
                        entropy_std = float("nan")
                    row["entropy_mean"] = float(entropy_mean)
                    row["entropy_std"] = float(entropy_std)
                    row["count"] = entropy_count

                    entropy_vals = region_stats["entropy_samples"]
                    if entropy_vals:
                        entropy_box_rows.extend(
                            {
                                "phase": phase_tag,
                                "epoch": int(epoch),
                                "stage": stage_name,
                                "region": region_name,
                                "entropy": float(v),
                            }
                            for v in entropy_vals
                        )

                    for d_name in labels:
                        cnt = int(region_stats["dir_count"].get(d_name, 0))
                        if cnt > 0:
                            mean_val = region_stats["dir_sum"][d_name] / cnt
                            var_val = max(
                                region_stats["dir_sq_sum"][d_name] / cnt - mean_val ** 2,
                                0.0,
                            )
                            std_val = math.sqrt(var_val)
                        else:
                            mean_val = float("nan")
                            std_val = float("nan")
                        row[f"{d_name}_mean"] = float(mean_val)
                        row[f"{d_name}_std"] = float(std_val)
                    gamma_rows.append(row)

            gamma_csv = self._experiment_file(f"{artifact_prefix}_gamma_stats.csv")
            entropy_csv = self._experiment_file(f"{artifact_prefix}_gamma_entropy_samples.csv")
            self._write_csv(gamma_csv, gamma_rows)
            self._write_csv(entropy_csv, entropy_box_rows)
            summary_payload["gamma_stats"] = {
                "stats_csv": os.path.basename(gamma_csv),
                "entropy_samples_csv": os.path.basename(entropy_csv),
            }

            if plt is not None and entropy_box_rows:
                # Plot boundary vs interior entropy distributions per stage.
                stages = sorted({r["stage"] for r in entropy_box_rows})
                fig = plt.figure(figsize=(4 * len(stages), 4))
                for idx, stg in enumerate(stages, start=1):
                    ax = fig.add_subplot(1, len(stages), idx)
                    b_vals = [
                        r["entropy"]
                        for r in entropy_box_rows
                        if r["stage"] == stg and r["region"] == "boundary"
                    ]
                    i_vals = [
                        r["entropy"]
                        for r in entropy_box_rows
                        if r["stage"] == stg and r["region"] == "interior"
                    ]
                    ax.boxplot([b_vals, i_vals], labels=["boundary", "interior"])
                    ax.set_title(stg)
                    ax.set_ylabel("Entropy")
                    ax.grid(True, linestyle="--", alpha=0.3)
                fig.tight_layout()
                fig.savefig(self._experiment_file(f"{artifact_prefix}_gamma_entropy_boxplot.png"), dpi=180)
                plt.close(fig)

        if save_base_artifacts or need_uncertainty_curve or need_gamma_stats:
            self._write_json(
                self._experiment_file(f"{artifact_prefix}_summary.json"),
                summary_payload,
            )

        self.last_eval_result = summary_payload

        return dice

    def validation(self, epoch):
        dice = self._evaluate(epoch, self.val_loader, phase="验证")
        val_loss = float(self.last_eval_result.get("loss", float("nan")))
        is_best = dice > self.best_pred
        if is_best:
            self.best_pred = dice

        cp = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_pred': self.best_pred,
            'best_dice': self.best_pred  # 保持一致性
        }
        if self.use_amp:
            cp['scaler'] = self.scaler.state_dict()
        self.saver.save_checkpoint(cp, is_best=is_best, filename='checkpoint_last.pth.tar')
        self.update_loss_history(
            epoch=epoch,
            val_loss=val_loss,
            val_dice=float(dice),
            is_best=is_best,
        )
        return {
            "dice": float(dice),
            "loss": val_loss,
            "is_best": bool(is_best),
        }

    def my_test(self, current_training_epoch):
        try:
            update_bn(self.train_loader, self.swa_model, device=self.device)
            self.model = self.swa_model.module
            self.model.eval()
            test_dice = self._evaluate(current_training_epoch, self.test_loader, phase="测试")

            # Evaluate EMA model in parallel with SWA reporting.
            original_model = self.model
            self.model = self.ema
            ema_test_dice = self._evaluate(current_training_epoch, self.test_loader, phase="测试-EMA")
            self.model = original_model

            return max(test_dice, ema_test_dice)
        except Exception as e:
            print(f"测试评估失败: {e}")
            traceback.print_exc()
            return 0.0

    def tta_predict(self, img, current_epoch=0):
        """针对MRI图像优化的测试时增强策略"""
        B, C, H, W = img.shape
        
        # 创建基本的几何变换TTA
        tta_imgs = [
            img,                                       # 原图
            torch.flip(img, dims=[-1]),                # 水平翻转
            torch.flip(img, dims=[-2]),                # 垂直翻转
            torch.rot90(img, k=1, dims=(-2, -1)),      # 旋转90度
        ]
        
        # 添加小角度旋转(更适合MRI图像的微小变化)
        angles = [5, -5, 10, -10]  # 小角度旋转更适合医学图像
        for angle in angles:
            # 使用插值旋转
            rad = angle * np.pi / 180
            cos_val, sin_val = np.cos(rad), np.sin(rad)
            rot_matrix = torch.tensor([[cos_val, -sin_val, 0], 
                                       [sin_val, cos_val, 0]], 
                                       device=img.device).float()
            grid = F.affine_grid(rot_matrix.repeat(B, 1, 1), img.size(), align_corners=False)
            rotated = F.grid_sample(img, grid, align_corners=False, mode='bilinear')
            tta_imgs.append(rotated)
        
        # 添加强度变换(针对MRI信号强度变化)
        intensity_factors = [0.95, 1.05]  # MRI信号强度微调
        for factor in intensity_factors:
            tta_imgs.append(img * factor)
        
        # 获取所有TTA变换的预测结果
        tta_preds = []
        for t in tta_imgs:
            with autocast(enabled=self.use_amp):
                logits = self._forward_model(t.to(self.device, non_blocking=True), current_epoch)
            tta_preds.append(torch.sigmoid(logits))
        
        # 还原变换和加权平均
        # 几何变换还原
        restored = [
            tta_preds[0],                                       # 原图
            torch.flip(tta_preds[1], dims=[-1]),                # 水平翻转恢复
            torch.flip(tta_preds[2], dims=[-2]),                # 垂直翻转恢复
            torch.rot90(tta_preds[3], k=3, dims=(-2, -1))       # 90度逆向旋转
        ]
        
        # 角度旋转还原
        for i, angle in enumerate(angles):
            rad = -angle * np.pi / 180  # 反向角度
            cos_val, sin_val = np.cos(rad), np.sin(rad)
            rot_matrix = torch.tensor([[cos_val, -sin_val, 0], 
                                       [sin_val, cos_val, 0]], 
                                       device=img.device).float()
            grid = F.affine_grid(rot_matrix.repeat(B, 1, 1), tta_preds[i+4].size(), align_corners=False)
            restored.append(F.grid_sample(tta_preds[i+4], grid, align_corners=False, mode='bilinear'))
        
        # 强度变换不需要还原
        for i in range(len(intensity_factors)):
            restored.append(tta_preds[i+4+len(angles)])
        
        # 加权平均 - 给原图和小角度旋转更高权重
        weights = [1.0] + [0.8] * 3 + [0.7] * len(angles) + [0.6] * len(intensity_factors)
        weighted_sum = 0
        weight_sum = sum(weights)
        
        for pred, weight in zip(restored, weights):
            weighted_sum += pred * weight
        
        avg = weighted_sum / weight_sum
        return torch.logit(avg.clamp(1e-6, 1 - 1e-6), eps=1e-6)

    def _resume_checkpoint(self, resume_path: str, finetune: bool):
        print(f"从检查点恢复: {resume_path}")
        try:
            cp = torch.load(resume_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(cp['state_dict'], strict=False)
            if missing:     print(f"  WARNING: 缺失的键: {missing}")
            if unexpected: print(f"  WARNING: 意外的键: {unexpected}")

            if not finetune:
                if 'optimizer' in cp: 
                    try:
                        self.optimizer.load_state_dict(cp['optimizer'])
                        print("  OK: 优化器状态已恢复")
                    except Exception as e: 
                        print(f"  WARNING: 优化器加载错误: {e}")
                
                if 'scheduler' in cp:
                    try: 
                        self.scheduler.load_state_dict(cp['scheduler'])
                        print("  OK: 调度器状态已恢复")
                    except Exception as e: 
                        print(f"  WARNING: 调度器加载错误: {e}")
                
                if self.use_amp and 'scaler' in cp:
                    try: 
                        self.scaler.load_state_dict(cp['scaler'])
                        print("  OK: AMP缩放器状态已恢复")
                    except Exception as e: 
                        print(f"  WARNING: 缩放器加载错误: {e}")
            else:
                print("  NOTE: 微调模式：优化器/调度器/缩放器不恢复")

            self.best_pred = cp.get('best_pred', self.best_pred)
            self.best_dice = cp.get('best_dice', getattr(self, 'best_dice', 0.0))
            self.start_epoch = cp.get('epoch', 0)
            print(f"  恢复 best_pred={self.best_pred:.4f}, best_dice={self.best_dice:.4f}, start_epoch={self.start_epoch}\n")
        except Exception as e:
            print(f"恢复检查点时出错: {e}")
            traceback.print_exc()



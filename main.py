import argparse
import inspect
import os
import traceback
from importlib import import_module
from pathlib import Path

import torch
import warnings

from train_code import Trainer, set_seed

warnings.filterwarnings("ignore")


def _resolve_model_class(model_name: str):
    net_root = Path(__file__).resolve().parent / "networks" / model_name
    if not net_root.exists():
        raise ImportError(f"Model directory not found: {net_root}")

    module_candidates = [f"networks.{model_name}.{model_name}"]
    for py_file in net_root.glob("*.py"):
        if py_file.stem.startswith("_"):
            continue
        module_candidates.append(f"networks.{model_name}.{py_file.stem}")

    checked = []
    for module_name in dict.fromkeys(module_candidates):
        try:
            module = import_module(module_name)
        except Exception:
            checked.append(module_name)
            continue
        cls = getattr(module, model_name, None)
        if isinstance(cls, type):
            return cls
        checked.append(module_name)

    raise ImportError(f"Cannot find class `{model_name}` in {checked}")


def _init_model_compat(model_cls, kwargs: dict):
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.values())
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    if accepts_var_kw:
        return model_cls(**kwargs), []

    accepted_keys = {p.name for p in params if p.name != "self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}
    dropped = [k for k in kwargs.keys() if k not in accepted_keys]
    return model_cls(**filtered_kwargs), dropped


def build_parser():
    parser = argparse.ArgumentParser(description="UMMNet main training")

    parser.add_argument("--dims", nargs=4, type=int, default=[48, 96, 192, 384])
    parser.add_argument("--blocks", nargs=4, type=int, default=[1, 1, 2, 1])
    parser.add_argument("--drop_paths", nargs=4, type=float, default=[0.1, 0.1, 0.2, 0.2])

    parser.add_argument("--dataset", type=str, default="pasd", choices=["pasd"])
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--t2_img_dir_rel", type=str, default="data/t2")
    parser.add_argument("--reg_bssfp_img_dir_rel", type=str, default="data/bssfp")
    parser.add_argument("--t2_mask_dir_rel", type=str, default="data/t2_mask")
    parser.add_argument("--reg_bssfp_mask_dir_rel", type=str, default="data/bssfp_mask")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--model", type=str, default="UMMNet", choices=["UMMNet"])
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--ummf_dilations", nargs=3, type=int, default=[1, 2, 4])
    parser.add_argument("--ummf_logvar_min", type=float, default=-8.0)
    parser.add_argument("--ummf_logvar_max", type=float, default=8.0)
    parser.add_argument("--lrf_msc_dilations", nargs=3, type=int, default=[1, 12, 24])
    parser.add_argument("--mc_dropout_p", type=float, default=0.2)
    parser.add_argument("--mc_dropout_samples_train", type=int, default=1)
    parser.add_argument("--mc_dropout_samples_eval", type=int, default=10)
    parser.add_argument("--udec_num_directions", type=int, default=4, choices=[4, 8])
    parser.add_argument("--udec_delta", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    pw_group = parser.add_mutually_exclusive_group()
    pw_group.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    pw_group.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./results_ummnet")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ft", action="store_true", default=False)
    parser.add_argument("--no_val", action="store_true", default=False)
    parser.add_argument("--skip_final_test", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--edge_kernel_size", type=int, default=9)
    parser.add_argument("--edge_multiplier", type=float, default=2.5)
    parser.add_argument("--edge_weight_min", type=float, default=1.0)
    parser.add_argument("--edge_weight_max", type=float, default=4.0)
    parser.add_argument("--dice_smooth", type=float, default=100.0)
    parser.add_argument("--dice_eps", type=float, default=1e-7)
    parser.add_argument("--dice_weight", type=float, default=0.75)
    parser.add_argument("--grad_clip_norm", type=float, default=2.0)
    parser.add_argument("--enable_batch_safety_checks", action="store_true", default=False)
    parser.add_argument("--snapshot_bn_for_recovery", action="store_true", default=False)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--bce_pos_weight_value", type=float, default=2.0)

    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--early_stop_min_delta", type=float, default=5e-4)
    parser.add_argument("--early_stop_min_epochs", type=int, default=30)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    fixed_seed = 42
    args.seed = fixed_seed
    args.split_seed = fixed_seed
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"

    model_cls = _resolve_model_class(args.model)
    model_kwargs = dict(
        num_classes=args.num_classes,
        dims=args.dims,
        blocks=args.blocks,
        drop_paths=args.drop_paths,
        ummf_dilations=args.ummf_dilations,
        ummf_logvar_min=args.ummf_logvar_min,
        ummf_logvar_max=args.ummf_logvar_max,
        lrf_msc_dilations=args.lrf_msc_dilations,
        mc_dropout_p=args.mc_dropout_p,
        mc_dropout_samples_train=args.mc_dropout_samples_train,
        mc_dropout_samples_eval=args.mc_dropout_samples_eval,
        udec_num_directions=args.udec_num_directions,
        udec_delta=args.udec_delta,
    )
    model, dropped = _init_model_compat(model_cls, model_kwargs)
    if dropped:
        print(f"ignored model args: {sorted(dropped)}")

    try:
        trainer = Trainer(args, model)
    except Exception as e:
        print(f"trainer init failed: {e}")
        traceback.print_exc()
        return

    if args.eval_only:
        if trainer.test_loader and hasattr(trainer, "_evaluate"):
            trainer.model.eval()
            dice = float(trainer._evaluate(trainer.start_epoch, trainer.test_loader, phase="test"))
            print(f"eval_only: test_dice={dice:.4f}")
        if getattr(trainer, "writer", None):
            trainer.writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    start_epoch_loop = trainer.start_epoch
    best_for_early_stop = float(getattr(trainer, "best_pred", 0.0))
    no_improve_epochs = 0
    patience = int(getattr(args, "early_stop_patience", 0))
    min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
    min_epochs = int(getattr(args, "early_stop_min_epochs", 0))
    early_stop_enabled = (patience > 0) and (not args.no_val) and bool(trainer.val_loader)

    for epoch in range(start_epoch_loop, args.epochs):
        try:
            train_loss = float(trainer.training(epoch))
            val_dice = float("nan")
            if not args.no_val and trainer.val_loader:
                val_result = trainer.validation(epoch)
                val_dice = float(val_result.get("dice", 0.0))
                if val_dice > (best_for_early_stop + min_delta):
                    best_for_early_stop = val_dice
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                if early_stop_enabled and (epoch + 1) >= min_epochs and no_improve_epochs >= patience:
                    print(
                        f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, "
                        f"val_dice={val_dice:.4f}, best_val={best_for_early_stop:.4f}, early_stop=1"
                    )
                    break
            if hasattr(trainer, "update_loss_history"):
                trainer.update_loss_history(epoch=epoch, train_loss=train_loss)
            print(
                f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, "
                f"val_dice={val_dice:.4f}, best_val={best_for_early_stop:.4f}"
            )
        except KeyboardInterrupt:
            print("training interrupted")
            break
        except Exception as e:
            print(f"training failed at epoch {epoch + 1}: {e}")
            traceback.print_exc()
            break

    if (not args.skip_final_test) and trainer.test_loader and hasattr(trainer, "my_test"):
        test_dice = float(trainer.my_test(args.epochs))
        print(f"final_test_dice={test_dice:.4f}")

    if getattr(trainer, "writer", None):
        trainer.writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

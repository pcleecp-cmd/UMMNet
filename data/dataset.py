import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def make_basic_aug(h: int, w: int):
    return A.Compose(
        [
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.15, contrast_limit=0.15),
            A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def make_advanced_aug(h: int, w: int):
    return A.Compose(
        [
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=20, alpha_affine=10, p=0.3),
            A.CoarseDropout(max_holes=5, max_height=32, max_width=32, p=0.2),
            A.RandomBrightnessContrast(p=0.15, contrast_limit=0.15),
            A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def _patient_id_from_stem(stem: str) -> str:
    m = re.match(r"(sub\d+|v\d+)", stem)
    return m.group(1) if m else stem.split("_")[0]


def _build_stem_map(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    stem_to_path: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        stem = p.stem
        if stem in stem_to_path:
            # Keep deterministic behavior when duplicate stems exist.
            continue
        stem_to_path[stem] = p
    return stem_to_path


class PASD_Dataset(Dataset):
    """
    Directory-driven PASD dataset.
    Expected folders are under project root (open):
    - data/t2
    - data/t2_mask
    - data/bssfp (or registered bSSFP dir)
    - data/bssfp_mask (or registered bSSFP mask dir)
    """

    def __init__(
        self,
        data_root: str = ".",
        split: str = "train",
        img_size=(224, 224),
        use_augmentation: bool = True,
        advanced_aug_epoch_threshold: int = 50,
        single_seq: int = 0,
        t2_img_dir_rel: str = "data/t2",
        bssfp_img_dir_rel: str = "data/bssfp",
        t2_mask_dir_rel: str = "data/t2_mask",
        bssfp_mask_dir_rel: str = "data/bssfp_mask",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        split_seed: Optional[int] = None,
    ):
        assert split in ("train", "val", "test", "all")
        assert single_seq in (0, 1, 2)

        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
        if abs(ratio_sum - 1.0) > 1e-6:
            train_ratio = train_ratio / ratio_sum
            val_ratio = val_ratio / ratio_sum
            test_ratio = test_ratio / ratio_sum

        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.single_seq = single_seq
        self.current_epoch = 0
        self.advanced_aug_epoch_threshold = advanced_aug_epoch_threshold
        self.use_augmentation = bool(use_augmentation and split == "train")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        self.t2_img_dir = self.data_root / t2_img_dir_rel
        self.bssfp_img_dir = self.data_root / bssfp_img_dir_rel
        self.t2_mask_dir = self.data_root / t2_mask_dir_rel
        self.bssfp_mask_dir = self.data_root / bssfp_mask_dir_rel

        if self.use_augmentation:
            h, w = self.img_size
            self.basic_aug = make_basic_aug(h, w)
            self.advanced_aug = make_advanced_aug(h, w)
        else:
            self.transform_val_test = A.Compose(
                [
                    A.Resize(
                        height=self.img_size[0],
                        width=self.img_size[1],
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True,
                    ),
                    A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5], max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets={"mask": "mask"},
            )

        self.samples = self._build_split_samples()
        print(
            f"[PASD_Dataset] split={self.split}, samples={len(self.samples)}, "
            f"ratios={self.train_ratio:.2f}:{self.val_ratio:.2f}:{self.test_ratio:.2f}, "
            f"split_seed={self.split_seed}"
        )

    def _build_split_samples(self) -> List[Dict[str, Path]]:
        t2_map = _build_stem_map(self.t2_img_dir)
        t2_mask_map = _build_stem_map(self.t2_mask_dir)
        bssfp_map = _build_stem_map(self.bssfp_img_dir)
        bssfp_mask_map = _build_stem_map(self.bssfp_mask_dir)

        overlap: Set[str] = (
            set(t2_map.keys())
            & set(t2_mask_map.keys())
            & set(bssfp_map.keys())
            & set(bssfp_mask_map.keys())
        )
        if not overlap:
            raise RuntimeError(
                "No overlapping samples found across 4 folders: "
                f"{self.t2_img_dir}, {self.t2_mask_dir}, {self.bssfp_img_dir}, {self.bssfp_mask_dir}"
            )

        all_samples: List[Dict[str, Path]] = []
        for stem in sorted(overlap):
            all_samples.append(
                {
                    "id": stem,
                    "patient_id": _patient_id_from_stem(stem),
                    "t2_img": t2_map[stem],
                    "t2_mask": t2_mask_map[stem],
                    "bssfp_img": bssfp_map[stem],
                    "bssfp_mask": bssfp_mask_map[stem],
                }
            )

        if self.split == "all":
            return all_samples

        patient_ids = sorted({item["patient_id"] for item in all_samples})
        if len(patient_ids) < 2:
            return all_samples

        train_val_ids, test_ids = train_test_split(
            patient_ids,
            test_size=self.test_ratio,
            random_state=self.split_seed,
            shuffle=True,
        )

        if len(train_val_ids) < 2:
            train_ids, val_ids = train_val_ids, []
        else:
            val_size_rel = self.val_ratio / max(self.train_ratio + self.val_ratio, 1e-12)
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=val_size_rel,
                random_state=self.split_seed,
                shuffle=True,
            )

        split_ids = {
            "train": set(train_ids),
            "val": set(val_ids),
            "test": set(test_ids),
        }
        keep_ids = split_ids[self.split]
        return [item for item in all_samples if item["patient_id"] in keep_ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample = self.samples[idx]
        file_identifier = sample["id"]
        try:
            t2_img_np = np.array(Image.open(sample["t2_img"]).convert("L"))
            bssfp_img_np = np.array(Image.open(sample["bssfp_img"]).convert("L"))
            mask_t2 = (np.array(Image.open(sample["t2_mask"]).convert("L")) > 0).astype(np.uint8)
            mask_bssfp = (np.array(Image.open(sample["bssfp_mask"]).convert("L")) > 0).astype(np.uint8)

            # Force bSSFP branch to match T2 spatial size.
            if bssfp_img_np.shape != t2_img_np.shape:
                h, w = t2_img_np.shape
                bssfp_img_np = cv2.resize(bssfp_img_np, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_bssfp = cv2.resize(mask_bssfp, (w, h), interpolation=cv2.INTER_NEAREST)

            combined_mask = np.maximum(mask_t2, mask_bssfp)
            multi_modal_img = np.stack((t2_img_np, bssfp_img_np), axis=-1)

            if self.single_seq == 1:
                multi_modal_img[..., 1] = 0
            elif self.single_seq == 2:
                multi_modal_img[..., 0] = 0

            if self.use_augmentation:
                use_basic = self.current_epoch < self.advanced_aug_epoch_threshold
                aug = (
                    self.basic_aug(image=multi_modal_img, mask=combined_mask)
                    if use_basic
                    else self.advanced_aug(image=multi_modal_img, mask=combined_mask)
                )
            else:
                aug = self.transform_val_test(image=multi_modal_img, mask=combined_mask)

            multi_modal_tensor = aug["image"]
            final_mask_tensor = aug["mask"].unsqueeze(0).float()
            final_mask_tensor = (final_mask_tensor > 0).float()
            return multi_modal_tensor, final_mask_tensor, {"id": file_identifier}

        except Exception as e:
            print(f"ERROR processing item {file_identifier} (Index {idx}): {e}")
            traceback.print_exc()
            return (
                torch.zeros((2, *self.img_size)),
                torch.zeros((1, *self.img_size)),
                {"id": f"ProcErr_{file_identifier}"},
            )

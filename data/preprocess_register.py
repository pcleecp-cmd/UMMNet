import argparse
from pathlib import Path
from typing import Dict, Set

import SimpleITK as sitk
from tqdm import tqdm


def build_stem_map(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    out: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.stem not in out:
            out[p.stem] = p
    return out


def remove_non_overlap(stem_map: Dict[str, Path], overlap: Set[str], dry_run: bool) -> int:
    removed = 0
    for stem, path in stem_map.items():
        if stem in overlap:
            continue
        if dry_run:
            print(f"[DRY-RUN] remove {path}")
        else:
            path.unlink(missing_ok=True)
        removed += 1
    return removed


def register_image_and_mask(
    fixed_img_path: Path,
    moving_img_path: Path,
    moving_mask_path: Path,
    output_img_path: Path,
    output_mask_path: Path,
) -> bool:
    try:
        fixed_image = sitk.ReadImage(str(fixed_img_path), sitk.sitkFloat32)
        moving_image = sitk.ReadImage(str(moving_img_path), sitk.sitkFloat32)
        moving_mask = sitk.ReadImage(str(moving_mask_path), sitk.sitkUInt8)

        transform = sitk.Euler2DTransform()
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, transform, sitk.CenteredTransformInitializerFilter.MOMENTS
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.6)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=80,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10,
            estimateLearningRate=registration_method.EachIteration,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.SetInitialTransform(initial_transform)

        final_transform = registration_method.Execute(fixed_image, moving_image)

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetReferenceImage(fixed_image)
        resample_image.SetInterpolator(sitk.sitkLinear)
        resample_image.SetDefaultPixelValue(0)
        resample_image.SetTransform(final_transform)
        registered_img = resample_image.Execute(moving_image)

        resample_mask = sitk.ResampleImageFilter()
        resample_mask.SetReferenceImage(fixed_image)
        resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        resample_mask.SetDefaultPixelValue(0)
        resample_mask.SetOutputPixelType(sitk.sitkUInt8)
        resample_mask.SetTransform(final_transform)
        registered_mask = resample_mask.Execute(moving_mask)

        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        registered_img_uint8 = sitk.Cast(sitk.RescaleIntensity(registered_img), sitk.sitkUInt8)
        sitk.WriteImage(registered_img_uint8, str(output_img_path))
        sitk.WriteImage(registered_mask, str(output_mask_path))
        return True
    except Exception as e:
        print(f"[ERROR] register failed for {moving_img_path.name}: {e}")
        return False


def main(args):
    data_root = Path(args.data_root)
    t2_dir = data_root / args.t2_dir_rel
    t2_mask_dir = data_root / args.t2_mask_dir_rel
    bssfp_dir = data_root / args.bssfp_dir_rel
    bssfp_mask_dir = data_root / args.bssfp_mask_dir_rel

    out_bssfp_dir = data_root / args.out_bssfp_dir_rel
    out_bssfp_mask_dir = data_root / args.out_bssfp_mask_dir_rel

    t2_map = build_stem_map(t2_dir)
    t2_mask_map = build_stem_map(t2_mask_dir)
    bssfp_map = build_stem_map(bssfp_dir)
    bssfp_mask_map = build_stem_map(bssfp_mask_dir)

    overlap = (
        set(t2_map.keys())
        & set(t2_mask_map.keys())
        & set(bssfp_map.keys())
        & set(bssfp_mask_map.keys())
    )
    print(
        f"[STATS] t2={len(t2_map)}, t2_mask={len(t2_mask_map)}, "
        f"bssfp={len(bssfp_map)}, bssfp_mask={len(bssfp_mask_map)}, overlap={len(overlap)}"
    )
    if not overlap:
        print("[STOP] no overlap found, nothing to process.")
        return

    if args.delete_non_overlap:
        removed = 0
        removed += remove_non_overlap(t2_map, overlap, args.dry_run)
        removed += remove_non_overlap(t2_mask_map, overlap, args.dry_run)
        removed += remove_non_overlap(bssfp_map, overlap, args.dry_run)
        removed += remove_non_overlap(bssfp_mask_map, overlap, args.dry_run)
        mode = "DRY-RUN" if args.dry_run else "APPLIED"
        print(f"[SYNC-{mode}] removed_non_overlap={removed}")

    if args.sync_only:
        print("[DONE] sync_only=True, registration skipped.")
        return

    out_bssfp_dir.mkdir(parents=True, exist_ok=True)
    out_bssfp_mask_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    for stem in tqdm(sorted(overlap), desc="register bSSFP to T2"):
        moving_img = bssfp_map[stem]
        moving_mask = bssfp_mask_map[stem]
        out_img = out_bssfp_dir / f"{stem}{args.registered_suffix}{moving_img.suffix}"
        out_mask = out_bssfp_mask_dir / f"{stem}{args.registered_suffix}{moving_mask.suffix}"

        if out_img.exists() and out_mask.exists() and not args.overwrite:
            continue

        ok = register_image_and_mask(
            fixed_img_path=t2_map[stem],
            moving_img_path=moving_img,
            moving_mask_path=moving_mask,
            output_img_path=out_img,
            output_mask_path=out_mask,
        )
        if ok:
            success += 1
        else:
            failed += 1

    print(f"[DONE] register_success={success}, register_failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Directory-driven preprocessing: overlap sync + bSSFP/T2 registration"
    )
    parser.add_argument("--data_root", type=str, default=".", help="project root (run under open)")

    parser.add_argument("--t2_dir_rel", type=str, default="data/t2")
    parser.add_argument("--t2_mask_dir_rel", type=str, default="data/t2_mask")
    parser.add_argument("--bssfp_dir_rel", type=str, default="data/bssfp")
    parser.add_argument("--bssfp_mask_dir_rel", type=str, default="data/bssfp_mask")

    parser.add_argument("--out_bssfp_dir_rel", type=str, default="data/bssfp_registered")
    parser.add_argument("--out_bssfp_mask_dir_rel", type=str, default="data/bssfp_mask_registered")
    parser.add_argument("--registered_suffix", type=str, default="_registered")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--delete_non_overlap", action="store_true", help="delete files not in 4-way overlap")
    parser.add_argument("--dry_run", action="store_true", help="for delete_non_overlap only")
    parser.add_argument("--sync_only", action="store_true", help="only sync, no registration")

    args = parser.parse_args()
    main(args)

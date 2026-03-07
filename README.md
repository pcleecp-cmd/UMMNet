# UMMNet

This folder contains only the main experiment code:

- `main.py`
- `train_code.py`
- `data/` (preprocessing + dataset loader only, no raw images)
- `kits/`
- `networks/UMMNet/`

## Notes

- Raw dataset images/masks are intentionally excluded.
- Ablation/rebuttal control options are removed from `main.py`.
- Training log output is simplified to one line per epoch.

## Run

```bash
python main.py --data_root . --save_dir ./results_ummnet
```

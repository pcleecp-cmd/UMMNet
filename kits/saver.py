# UMMNet-main/UMMNet/kits/saver.py
# Modified to remove dependency on args.modalities

import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):
        self.args = args
        # --- MODIFIED Directory Path ---
        # Original line used args.modalities:
        # self.directory = os.path.join(args.save_dir, args.dataset, args.modalities + "_" + args.model)
        # New line uses only dataset and model:
        self.directory = os.path.join(args.save_dir, args.dataset, args.model)
        # --- End MODIFICATION ---

        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        # Original run_id formatting: 'experiment_{:0>3d}'. Keeping it for consistency.
        print(f'Saver: Assigning run_id: {run_id}')

        self.experiment_dir = os.path.join(self.directory, 'experiment_{:0>3d}'.format(run_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            print(f"Saver: Created experiment directory: {self.experiment_dir}")
        else:
             print(f"Saver: Using existing experiment directory: {self.experiment_dir}")


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filepath = os.path.join(self.experiment_dir, filename)
        torch.save(state, filepath)
        if is_best:
            best_pred = state['best_pred'] # Assuming 'best_pred' key exists in state
            best_pred_filepath = os.path.join(self.experiment_dir, 'best_pred.txt')
            model_best_path = os.path.join(self.experiment_dir, 'model_best.pth.tar') # Consistent best model name
            try:
                with open(best_pred_filepath, 'w') as f:
                    f.write(f"{best_pred:.6f}") # Write with precision

                # Simplified logic: always copy if it's the best for this run
                shutil.copyfile(filepath, model_best_path)
                print(f"Saver: Saved best model checkpoint (Score: {best_pred:.4f}) to {model_best_path}")

            except Exception as e:
                 print(f"Error saving best model artifacts: {e}")


    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        try:
            log_file = open(logfile, 'w', encoding='utf-8')
            p = OrderedDict(sorted(vars(self.args).items(), key=lambda kv: kv[0]))
            print("Saver: Saving experiment config:")
            for key, val in p.items():
                log_file.write(f"{key}:{val}\n")
                print(f"  {key}: {val}")
            log_file.close()
        except Exception as e:
             print(f"Error saving experiment config to {logfile}: {e}")

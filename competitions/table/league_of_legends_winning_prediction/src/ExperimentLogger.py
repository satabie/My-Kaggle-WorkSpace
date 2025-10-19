import yaml
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import os


class ExperimentLogger:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.exp_name = self.config["experiment"]["name"]
        self.version = self.config["experiment"]["version"]

    def log_experiment(self, cv_valid_auc, cv_valid_acc, oof_auc, oof_acc, best_iterations=None):
        result = {
            "experiment": f"{self.exp_name}_v{self.version}",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": self.config.get("experiment", {}).get("description", ""),
            # スコア - AUC
            "oof_auc": float(oof_auc),
            "cv_auc_mean": float(np.mean(cv_valid_auc)),
            "cv_auc_std": float(np.std(cv_valid_auc)),
            "cv_auc_per_fold": [float(x) for x in cv_valid_auc],
            # スコア - Accuracy
            "oof_acc": float(oof_acc),
            "cv_acc_mean": float(np.mean(cv_valid_acc)),
            "cv_acc_std": float(np.std(cv_valid_acc)),
            "cv_acc_per_fold": [float(x) for x in cv_valid_acc],
            # パラメータ
            "params": self.config.get("model", {}).get("lgbm", {}),
            "threshold": self.config.get("threshold", {}).get("optimal", 0.5),
        }

        # オプション: Best iterations
        if best_iterations is not None:
            result["best_iterations"] = [int(x) for x in best_iterations]
            result["mean_iteration"] = float(np.mean(best_iterations))

        # JSONで保存（ディレクトリが存在しない場合は作成）
        output_dir = "../outputs/experiments"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"{self.exp_name}_v{self.version}.json")
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Experiment: {self.exp_name}_v{self.version}")
        print(f"OOF AUC: {oof_auc:.4f} | Accuracy: {oof_acc:.4f}")
        print(f"CV AUC:  {np.mean(cv_valid_auc):.4f} ± {np.std(cv_valid_auc):.4f}")
        print(f"CV ACC:  {np.mean(cv_valid_acc):.4f} ± {np.std(cv_valid_acc):.4f}")
        print(f"Saved to: {filename}")
        print(f"{'='*50}\n")

import json
import csv
import os
from pathlib import Path
from datetime import datetime

class FlexibleLogger:
    def __init__(self, config, model_name: str, dataset_name: str, split_method: str):
        """
        Args:
            config: 日志配置
            model_name: 模型名称 (e.g. "EEGNet")
            dataset_name: 数据集名称 (e.g. "EEGDataset_DTU")
            split_method: 划分方式 (e.g. "random")
        """
        self.config = config["logging"]
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split_method = split_method
        
        # 生成文件名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self._generate_filename()
        
        # 创建目录
        os.makedirs(self.config["save_dir"], exist_ok=True)
        self.filepath = Path(self.config["save_dir"]) / self.filename
        self._init_file()

    def _generate_filename(self) -> str:
        """生成动态文件名"""
        return self.config["filename_template"].format(
            model=self.model_name,
            dataset=self.dataset_name,
            split=self.split_method,
            timestamp=self.timestamp
        ) + self._get_extension()

    def _get_extension(self) -> str:
        """根据格式返回文件后缀"""
        return {
            "json": ".json",
            "csv": ".csv",
            "txt": ".log"
        }[self.config["format"]]

    def _init_file(self):
        """初始化日志文件"""
        if self.config["format"] == "json":
            with open(self.filepath, "w") as f:
                json.dump({"metadata": {
                    "model": self.model_name,
                    "dataset": self.dataset_name,
                    "split": self.split_method,
                    "start_time": self.timestamp
                }}, f, indent=2)
        elif self.config["format"] == "csv":
            with open(self.filepath, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "metric", "value"])

    def log(self, metrics: dict, epoch: int):
        """记录指标"""
        if self.config["format"] == "json":
            self._log_json(metrics, epoch)
        elif self.config["format"] == "csv":
            self._log_csv(metrics, epoch)
        else:  # txt
            self._log_text(metrics, epoch)

    def _log_json(self, metrics: dict, epoch: int):
        with open(self.filepath, "r+") as f:
            data = json.load(f)
            data[f"epoch_{epoch}"] = metrics
            f.seek(0)
            json.dump(data, f, indent=2)

    def _log_csv(self, metrics: dict, epoch: int):
        with open(self.filepath, "a", newline='') as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                writer.writerow([epoch, k, v])

    def _log_text(self, metrics: dict, epoch: int):
        with open(self.filepath, "a") as f:
            f.write(f"[Epoch {epoch}] " + " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
            ) + "\n")
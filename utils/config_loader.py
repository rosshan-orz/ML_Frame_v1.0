# utils/config_loader.py
import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import importlib
from torch.utils.data import Dataset

def load_dataset_class(dataset_class_name: str) -> type[Dataset]:
    """
    动态加载数据集类
    Args:
        dataset_class_name: 格式为"module.submodule.ClassName" 或 "ClassName"（默认从.datasets导入）
    Returns:
        数据集类（未实例化）
    Raises:
        ImportError: 类不存在时抛出异常
    """
    try:
        from dataset import eeg_dataset  # 数据集定义在./dataset/eeg_dataset.py中，封装成类，继承自torch.utils.data.Dataset
        module = eeg_dataset
        class_name = dataset_class_name
        
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to load dataset class '{dataset_class_name}'. "
            f"Original error: {str(e)}"
        ) from e

class ConfigLoader:
    """动态加载和验证YAML配置文件"""
    
    @staticmethod
    def load_config(yaml_path: str) -> Dict[str, Any]:
        """
        加载YAML配置并执行基础验证
        Args:
            yaml_path: 配置文件路径
        Returns:
            配置字典（包含自动生成的绝对路径）
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 关键字段缺失
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 自动补全路径为绝对路径
        if "data" in config and "root" in config["data"]:
            config["data"]["root"] = str(config["data"]["root"])
        
        # 基础验证
        required_sections = {
            "task": ["name"],
            "data": ["root", "file", "total_sub"],
            "training": ["batch_size", "lr"]
        }
        for section, fields in required_sections.items():
            if section not in config:
                raise ValueError(f"Missing section: {section}")
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing field '{field}' in section '{section}'")
        
        """
        # 设置默认值
        config["training"].setdefault("weight_decay", 0.001)
        config["training"].setdefault("patience", 1)
        config["training"].setdefault("factor", 0.9)
        config["training"].setdefault("temperature", 0.1)
        config["environment"].setdefault("device", "auto")
        config["environment"].setdefault("save_dir", "./results")
        """
        
        return config
    
    @staticmethod
    def _resolve_path(base_path: Path, target_path: str) -> Path:
        """解析相对路径为绝对路径"""
        path = (base_path / target_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path


class RuntimeConfig:
    """运行时配置容器"""
    
    def __init__(self, config_dict: Dict[str, Any], dataset_class: type):
        """
        Args:
            config_dict: 从YAML加载的配置字典
            dataset_class: 数据集类（需在运行时传入）
        """
        self._config = config_dict
        self.subject = "1"  # 默认初始值
        self.dataset_class = load_dataset_class(
            config_dict["data"]["dataset_class"]
        )
        
        # 自动创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
    
    @property
    def file_name(self) -> str:
        """动态生成文件名（使用当前subject）"""
        return self._config["data"]["file"].format(subject=self.subject)
    
    @property
    def device(self) -> torch.device:
        """自动检测设备"""
        device = self._config["environment"]["device"]
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    # 常用属性的快捷访问
    @property
    def root(self) -> str: return self._config["data"]["root"]
    
    @property
    def total_sub(self) -> int: return self._config["data"]["total_sub"]
    
    @property
    def batch_size(self) -> int: return self._config["training"]["batch_size"]
    
    @property
    def lr(self) -> float: return self._config["training"]["lr"]
    
    @property
    def epochs(self) -> int: return self._config["training"]["epochs"]
    
    @property
    def save_dir(self) -> str: return self._config["environment"]["save_dir"]
    
    @property
    def num_channels(self) -> int: return self._config["data"]["num_channels"]
    
    def __getitem__(self, key):
        """支持深度字典访问，如 config['data']['root'] 或 config['data.root']"""
        try:
            if '.' in key:
                # 处理 config['data.root'] 这种形式
                keys = key.split('.')
                value = self._config
                for k in keys:
                    value = value[k]
                return value
            else:
                # 处理常规 config['data'] 这种形式
                return self._config[key]
        except KeyError as e:
            raise KeyError(f"Config key '{key}' not found. Available keys: {list(self._config.keys())}") from e
    
    def __contains__(self, key):
        """支持in操作符，检查键是否存在"""
        if '.' in key:
            # 处理嵌套键的情况
            keys = key.split('.')
            value = self._config
            for k in keys:
                if k not in value:
                    return False
                value = value[k]
            return True
        return key in self._config

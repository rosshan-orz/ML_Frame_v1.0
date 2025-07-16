import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseTask(ABC):
    """所有任务类型的抽象基类"""
    def __init__(self, config: 'Config'):
        """
        Args:
            config: 全局配置对象
        """
        self.config = config
        self.device = config.device
        self._train_dataset = None
        self._val_dataset = None

    # ------------------- 必须实现的抽象方法 -------------------
    @abstractmethod
    def _load_raw_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """初始化模型实例"""
        pass

    @abstractmethod
    def train_step(self, 
                 batch: Tuple[torch.Tensor, ...], 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer) -> float:
        """单批次训练逻辑
        Returns:
            当前batch的loss值
        """
        pass

    @abstractmethod
    def valid_step(self, 
                val_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module) -> Dict[str, float]:
        """完整验证逻辑
        Returns:
            指标字典 (e.g. {"accuracy": 0.95, "loss": 0.1})
        """
        pass

    # ------------------- 可选公共方法 -------------------
    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        # 加载完整数据集
        full_dataset = self._load_raw_dataset()  # 需实现具体加载逻辑

        # 获取划分策略
        splitter = get_splitter(self.config.data["split_method"])
        train_set, val_set = splitter.split(
            full_dataset,
            **self.config.data.get("split_params", {})
        )
    
        # 可选: 数据增强
        if self.config.data.get("augment_train", False):
            train_set = AugmentedDataset(train_set)  # 自定义增强包装器
            
        return train_set, val_set
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建标准数据加载器"""
        if self._train_dataset is None:
            self._train_dataset, self._val_dataset = self.create_datasets()
        
        train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False
        )
        return train_loader, val_loader
    
    def compute_metrics(self, input_dict: Dict) -> Dict[str, float]:
        """
        计算任务相关Metrics
        默认使用全局注册的Metrics，子类可覆盖
        """
        return MetricRegistry.compute(input_dict)
from typing import Dict, Type, Tuple
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
import numpy as np
from abc import ABC, abstractmethod

class BaseSplitStrategy(ABC):
    """划分策略基类"""
    @abstractmethod
    def split(self, dataset: Dataset, **params) -> Tuple[Subset, Subset]:
        pass

# 具体策略实现
class RandomSplitStrategy(BaseSplitStrategy):
    def split(self, dataset, **params):
        train_len = int(len(dataset) * params["train_ratio"])
        valid_len = int(len(dataset) - train)
        Train_dataset, Valid_dataset = random_split(dataset, [train_len, val_len])
        
        return Train_dataset, Valid_dataset

class OrderSplitStrategy(BaseSplitStrategy):
    def split(self, dataset, **params):
        train_len = int(len(dataset) * params["train_ratio"])
        valid_len = int(len(dataset) - train)
        Train_dataset = Subset(TotalDataset, range(train_len))
        Valid_dataset = Subset(TotalDataset, range(train_len, total_len))
        
        return Train_dataset, Valid_dataset
    
# 策略注册表
_SPLIT_STRATEGIES: Dict[str, Type[BaseSplitStrategy]] = {
    "random": RandomSplitStrategy,
    "order": OrderSplitStrategy,
    # 可扩展其他策略...
}

def get_splitter(method: str) -> BaseSplitStrategy:
    """获取划分策略实例"""
    if method not in _SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split method: {method}. "
                       f"Available: {list(_SPLIT_STRATEGIES.keys())}")
    return _SPLIT_STRATEGIES[method]()
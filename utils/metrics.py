import torch
import torch.nn.functional as F
from typing import Dict, Callable, Any, Optional
from functools import partial

class MetricRegistry:
    """Metrics集中注册与管理类"""
    
    _registry = {}  # 全局注册表
    
    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        """注册一个新的metric计算函数"""
        if name in cls._registry:
            raise ValueError(f"Metric '{name}' already registered")
        cls._registry[name] = func
    
    @classmethod
    def compute(
        cls,
        input_dict: Dict[str, Any],
        metrics: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        批量计算Metrics
        Args:
            input_dict: 包含logits, labels, loss等字段的字典
            metrics: 指定要计算的metrics列表，为None时计算所有注册项
        Returns:
            指标字典 {metric_name: value}
        """
        results = {}
        metrics = metrics or list(cls._registry.keys())
        
        for name in metrics:
            if name not in cls._registry:
                raise KeyError(f"Metric '{name}' not registered. Available: {list(cls._registry.keys())}")
            results[name] = cls._registry[name](input_dict)
            
        return results

# ================== 内置Metrics实现 ==================

def _get_required(input_dict: Dict, *keys) -> Any:
    """安全获取字典中的必需字段"""
    for key in keys:
        if key not in input_dict:
            raise KeyError(f"Input dict missing required key: '{key}'")
    return (input_dict[k] for k in keys)

def accuracy(input_dict: Dict) -> float:
    """计算分类准确率"""
    logits, labels = _get_required(input_dict, 'logits', 'labels')
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def cross_entropy_loss(input_dict: Dict) -> torch.Tensor:
    logits, labels = _get_required(input_dict, 'logits', 'labels')
    return F.cross_entropy(logits, labels)

# ================== 注册Metrics ==================
MetricRegistry.register('accuracy', accuracy)
MetricRegistry.register('loss', cross_entropy_loss)

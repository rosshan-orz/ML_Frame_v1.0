# my_ml_framework/model/__init__.py
import importlib
import torch.nn as nn
from typing import Dict, Type

# 预置模型注册表（可扩展）
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """装饰器注册新模型"""
    def decorator(cls: Type[nn.Module]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    根据配置动态获取模型实例
    Args:
        model_name: 模型类名或路径 (e.g. "EEGNet" 或 "external_module.CustomModel")
        **kwargs: 模型初始化参数
    Returns:
        初始化后的模型实例
    Raises:
        ImportError: 模型类不存在时抛出
    """
    try:
        # 情况1: 从注册表直接获取
        if model_name in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[model_name](**kwargs)
        
        # 情况2: 尝试从本地models目录导入
        try:
            module = importlib.import_module(f"model.{model_name.lower()}")
            return getattr(module, model_name)(**kwargs)
        except ImportError:
            raise ValueError(f"Model '{model_name}' not found in registry or local modules")
    except Exception as e:
        raise ImportError(f"Failed to load model '{model_name}': {str(e)}")
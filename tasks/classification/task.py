import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict
from core.base_task import BaseTask
from utils.metrics import MetricRegistry
from utils.config_loader import RuntimeConfig, load_dataset_class
from models import get_model  # 工厂方法导入

class BinaryClassificationTask(BaseTask):
    """
    二分类任务实现类
    继承自BaseTask，实现抽象方法和通用逻辑
    """

    def __init__(self, config: RuntimeConfig):
        """
        初始化二分类任务
        
        Args:
            config: 运行时配置对象
        """
        super().__init__(config)
        
        # 可以在这里初始化其他需要的组件

    @property
    def _load_raw_dataset(self) -> Dataset:
        """
        加载原始数据集
        实现从磁盘加载原始数据集的逻辑
        
        Returns:
            返回原始数据集对象
        """
        # 从配置中获取数据集参数
        data_config = self.config["data"]
        
        # 动态加载数据集类
        dataset_class = load_dataset_class(data_config["dataset_class"])
        dataset = dataset_class(
            root=data_config["root"],
            file_name=data_config["file"].format(self.config.subject)  # 使用默认subject为1的数据
        )
        # 创建数据集实例
        return dataset

    def create_model(self) -> torch.nn.Module:
        """
        创建模型实例
        
        Returns:
            返回模型实例
        """
        return get_model(
            model_name=self.config["model"]["class_name"],
            num_channels=self.config["data"]["num_channels"]
        ).to(self.device)

    def train_step(self, 
                 batch: tuple[torch.Tensor, ...], 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer) -> Tuple[float, Dict]:
        """
        单批次训练逻辑
        
        Args:
            batch: 包含输入数据和标签的元组
            model: 要训练的模型
            optimizer: 优化器
            
        Returns:
            当前batch的loss值和指标字典
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.float().to(self.device).unsqueeze(1)  # 添加通道维度
        
        # 前向传播
        outputs = model(inputs)
        
        # 通过metrics接口获取loss
        metrics = self.compute_metrics({
            'logits': outputs.detach(),
            'labels': labels.long()
        })
        loss = metrics['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        return loss.item(), metrics

    def valid_step(self, 
                batch: tuple[torch.Tensor, ...],
                model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        单批次验证逻辑
        
        Args:
            batch: 包含输入数据和标签的元组
            model: 要验证的模型
            
        Returns:
            返回指标字典
        """
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.float().to(self.device).unsqueeze(1)  # 添加通道维度
        
        with torch.no_grad():
            outputs = model(inputs)
            # 通过metrics接口获取loss
            metrics = self.compute_metrics({
                'logits': outputs.detach(),
                'labels': labels.long()
            })
            logits = outputs.detach()
        
        return logits, labels, metrics
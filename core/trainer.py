# core/trainer.py
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from utils.config_loader import RuntimeConfig

class Trainer:
    def __init__(
        self,
        task,  # BaseTask实例
        config: RuntimeConfig,
        logger=None,
        callbacks: Optional[list] = None
    ):
        """
        Args:
            task: 具体任务实例
            config: 全局配置字典
            logger: 日志记录器
            callbacks: 回调函数列表
        """
        self.task = task
        self.config = config['training']
        self.logger = logger
        self.callbacks = callbacks or []
        
        # 初始化核心组件
        self.model = task.create_model().to(task.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 训练状态跟踪
        self.epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """根据配置创建优化器"""
        optim_map = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'adamw': torch.optim.AdamW
        }
        return optim_map[self.config['optimizer']](
            self.model.parameters(),
            lr=self.config['lr'],
            **self.config.get('optimizer_params', {})
        )

    def _create_scheduler(self):
        """创建学习率调度器"""
        if 'scheduler' not in self.config:
            return None
            
        schedulers = {
            'step': torch.optim.lr_scheduler.StepLR,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
        }
        return schedulers[self.config['scheduler']](
            self.optimizer,
            **self.config['scheduler_params']
        )

    def run(self):
        """主训练循环"""
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            self._run_epoch()
            
            # 验证阶段
            if epoch % self.config['eval_freq'] == 0:
                val_metrics = self._validate()
                self._checkpoint(val_metrics)
                
            # 回调处理
            for callback in self.callbacks:
                callback(self)

    def _run_epoch(self):
        """执行单个epoch训练"""
        self.model.train()
        pbar = tqdm(self.task.get_data_loaders()[0], desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # 训练步骤
            loss, metrics = self.task.train_step(batch, self.model, self.optimizer)
            
            # 日志记录
            if self.logger:
                self.logger.log({
                    'train/loss': loss,
                    **metrics,
                    'lr': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
            
            # 更新进度条
            pbar.set_postfix(loss=loss)
            self.global_step += 1

        # 更新学习率
        if self.scheduler:
            self.scheduler.step()

    def _validate(self) -> Dict[str, float]:
        """验证流程"""
        self.model.eval()
        val_loader = self.task.get_data_loaders()[1]
        all_metrics = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                _logits, _labels, metrics = self.task.valid_step(batch, self.model)
                # 确保设备一致性
                all_logits.append(_logits)
                all_labels.append(_labels)
                all_metrics.append(metrics)
        
        # 聚合指标
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0]
        }
        
        if self.logger:
            self.logger.log(
                {f'val/{k}': v for k, v in avg_metrics.items()},
                step=self.global_step
            )
        
        return avg_metrics

    def _checkpoint(self, metrics: Dict[str, float]):
        """模型保存逻辑"""
        current_metric = metrics[self.config['monitor_metric']]
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            
            checkpoint = {
                'epoch': self.epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }
            
            torch.save(
                checkpoint,
                Path(self.config['save_dir']) / f"best_model.pt"
            )
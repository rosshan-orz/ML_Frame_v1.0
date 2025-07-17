# TODO
# utils/callbacks.py
from typing import Any, Dict, List, Optional
import torch
from pathlib import Path

class Callback:
    """所有回调的抽象基类"""
    def on_init_end(self, trainer):
        """训练器初始化完成后调用"""
        pass

    def on_train_start(self, trainer):
        """训练开始前调用"""
        pass

    def on_train_end(self, trainer):
        """训练结束后调用"""
        pass

    def on_epoch_start(self, trainer):
        """每个epoch开始时调用"""
        pass

    def on_epoch_end(self, trainer):
        """每个epoch结束时调用"""
        pass

    def on_batch_start(self, trainer):
        """每个batch开始时调用"""
        pass

    def on_batch_end(self, trainer):
        """每个batch结束时调用"""
        pass

    def on_validation_start(self, trainer):
        """验证开始前调用"""
        pass

    def on_validation_end(self, trainer):
        """验证结束后调用"""
        pass

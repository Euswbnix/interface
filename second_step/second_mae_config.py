# config.py
import os
import torch
import numpy as np
import random


class MaeSecondConfig:
    # 数据配置
    train_root = './train'
    val_root = './val'
    train_list = './train.txt'
    val_list = './val.txt'

    # 类别映射
    class_names = [
        'biyan',
        'normal',
        'zhuyili_no_focus',
    ]

    # 模型配置
    model_name = 'MCG-NJU/videomae-base'  # VideoMAE-Base
    # 可选: 'MCG-NJU/videomae-large' (需要更多显存)
    num_classes = 3

    # 视频采样配置
    num_frames = 16  # VideoMAE 使用16帧
    sampling_rate = 4  # 采样间隔 (16帧 × 4 = 64帧覆盖)
    target_size = (224, 224)  # VideoMAE 使用224x224

    # 训练配置
    batch_size = 8  # V100 32G 可以设置8-16
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.05

    # 优化器配置
    warmup_epochs = 5
    lr_scheduler = 'cosine'

    # 数据增强
    use_temporal_jitter = True  # 时间抖动
    use_horizontal_flip = True  # 水平翻转（可以适当使用）
    use_color_jitter = True  # 颜色抖动

    # 设备配置
    device = 'cuda'
    num_workers = 4
    pin_memory = True
    mixed_precision = True  # 混合精度训练

    # 梯度累积
    gradient_accumulation_steps = 2

    # 推理配置
    inference_batch_size = 16
    num_segments = 1  # 推理时分成几个片段

    # 随机种子
    seed = 42

    def __init__(self):
        self._set_seed()
        self._print_config()

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        print(f"随机种子已设置: {self.seed}")

    def _print_config(self):
        print("=" * 60)
        print("VideoMAE 训练配置")
        print("=" * 60)
        print(f"类别数: {self.num_classes}")
        for idx, name in enumerate(self.class_names):
            print(f"  {idx}: {name}")
        print(f"\n模型: {self.model_name}")
        print(f"采样: {self.num_frames}帧, 间隔{self.sampling_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"学习率: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print("=" * 60)

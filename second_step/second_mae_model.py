# model.py
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import os
class VideoMAClassifier(nn.Module):
    """VideoMAE 分类模型"""
    
    def __init__(self, 
                 model_name='MCG-NJU/videomae-base',
                 num_classes=8,
                 freeze_backbone=False):
        super().__init__()
        
        print(f"加载 VideoMAE 模型: {model_name}")

        local_model_path = '/video/interface/videomae-base-local'

        # 加载预训练模型
        if os.path.exists(local_model_path):
            self.model = VideoMAEForVideoClassification.from_pretrained(
                local_model_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # 加载预训练模型
            self.model = VideoMAEForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )

        
        # 可选：冻结主干
        if freeze_backbone:
            print("冻结 VideoMAE 主干网络")
            for param in self.model.videomae.parameters():
                param.requires_grad = False
        
        # 获取特征维度
        self.feature_dim = self.model.config.hidden_size
        print(f"特征维度: {self.feature_dim}")
        
        if os.path.exists(local_model_path):
            # 图像处理器（用于预处理）
            self.image_processor = VideoMAEImageProcessor.from_pretrained(local_model_path)
        else:
            self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name)

    def forward(self, pixel_values, labels=None):
        """
        Args:
            pixel_values: [batch, C, T, H, W] 范围 [0,1]
        Returns:
            logits: [batch, num_classes]
        """
        # 确保输入格式正确
        if pixel_values.dim() == 5:
            # [batch, C, T, H, W] -> [batch, T, C, H, W]
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        # 前向传播
        outputs = self.model(pixel_values=pixel_values)
        
        return outputs.logits
    
    def extract_features(self, pixel_values):
        """提取视频特征"""
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        outputs = self.model.videomae(pixel_values)
        
        # 取 [CLS] token 的特征
        features = outputs.last_hidden_state[:, 0, :]
        
        return features

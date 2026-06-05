# inference.py
import torch
import numpy as np
import cv2
import os
from pathlib import Path
from second_step.second_mae_config import MaeSecondConfig
from second_step.second_mae_model import VideoMAClassifier


class VideoMAEInference:
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        self.model = VideoMAClassifier(
            model_name=config.model_name,
            num_classes=config.num_classes
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"加载模型: {checkpoint_path}")
        print(f"最佳验证准确率: {checkpoint.get('best_acc', 0):.2f}%")

    def preprocess_video(self, frames_raw):

        if len(frames_raw) == 0 or len(frames_raw) < 64:
            print('frame len invalid')
            return None

        frames = []

        for frame in frames_raw:
            temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, self.config.target_size)
            frames.append(temp)

        frames = np.array(frames, dtype=np.float32) / 255.0

        # 多片段采样
        required_frames = self.config.num_frames * self.config.sampling_rate
        num_segments = self.config.num_segments
        segment_len = required_frames // num_segments

        segments = []
        for i in range(num_segments):
            start = i * segment_len
            end = start + segment_len
            segment = frames[start:end] if end <= len(frames) else frames[start:]

            # 采样
            indices = list(range(0, len(segment), self.config.sampling_rate))
            sampled = segment[indices]

            if len(sampled) < self.config.num_frames:
                pad = self.config.num_frames - len(sampled)
                sampled = np.concatenate([sampled, [sampled[-1]] * pad], axis=0)
            else:
                sampled = sampled[:self.config.num_frames]

            # 转换为 [C, T, H, W]
            sampled = torch.from_numpy(sampled).float()
            sampled = sampled.permute(3, 0, 1, 2)
            segments.append(sampled)

        return segments

    def predict(self, frames_raw):
        """预测单个视频"""
        segments = self.preprocess_video(frames_raw)
        if segments is None:
            return None

        segment_logits = []
        with torch.no_grad():
            for segment in segments:
                segment = segment.unsqueeze(0).to(self.device)
                outputs = self.model(segment)
                segment_logits.append(outputs.cpu())

        # 融合
        avg_logits = torch.mean(torch.stack(segment_logits), dim=0)
        probs = torch.softmax(avg_logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        return {
            'class_id': pred_class,
            'class_name': self.config.class_names[pred_class],
            'confidence': probs[0][pred_class].item(),
            #'probabilities': {name: probs[0][i].item() for i, name in enumerate(self.config.class_names)}
        }


def main():
    config = MaeSecondConfig()
    inferencer = VideoMAEInference('checkpoints/best_model.pth', config)

    # 测试单个视频
    test_video = 'test.mp4'
    if os.path.exists(test_video):
        result = inferencer.predict(test_video)
        print(f"\n预测结果: {result['class_name']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n各类别概率:")
        for name, prob in result['probabilities'].items():
            print(f"  {name}: {prob:.4f}")


if __name__ == '__main__':
    main()

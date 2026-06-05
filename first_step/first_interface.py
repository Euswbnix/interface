import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tempfile
import logging

import time

# MMAction2 相关导入
from mmaction.apis import inference_recognizer, init_recognizer
from first_step.first_mae_interface import VideoMAEInference
from first_step.first_mae_config import MaeConfig

class FirstSliceInference:
    """视频切片行为识别推理类 - 每个切片单独输出结果"""

    def __init__(self, slow_model_configs, mae_model_configs, device='cuda:0', clip_len=64, overlap=20, logger=None):
        """
        初始化多个模型

        Args:
            model_configs: 列表，每个元素为 (config_path, checkpoint_path, model_name)
            mae_model_configs: 列表，每个元素为 (checkpoint_path, model_name)
            device: 推理设备
            clip_len: 每个片段长度
            overlap: 滑动窗口重叠帧数
            logger: 日志记录器
        """
        self.device = device
        self.clip_len = clip_len
        self.overlap = overlap
        self.stride = clip_len - overlap  # 滑动步长
        self.logger = logger or logging.getLogger()

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix='mmaction_inference_')

        # 加载多个模型
        self.models = []
        self.model_names = []
        self.model_configs = []

        for i, model_info in enumerate(slow_model_configs):
            # 处理不同长度的元组
            if len(model_info) == 3:
                config_path, checkpoint_path, model_name = model_info
            else:
                config_path, checkpoint_path = model_info
                model_name = f"Model_{i + 1}_Epoch{Path(checkpoint_path).stem.split('_')[-1]}"

            print(f"\n加载模型 {i + 1}: {model_name}")
            print(f"  配置文件: {config_path}")
            print(f"  权重文件: {checkpoint_path}")

            # 检查文件是否存在
            if not Path(config_path).exists():
                error_msg = f"配置文件不存在: {config_path}"
                print(f"  错误: {error_msg}")
                raise FileNotFoundError(error_msg)
            if not Path(checkpoint_path).exists():
                error_msg = f"权重文件不存在: {checkpoint_path}"
                print(f"  错误: {error_msg}")
                raise FileNotFoundError(error_msg)

            try:
                model = init_recognizer(config_path, checkpoint_path, device=device)
                self.models.append(model)
                self.model_names.append(model_name)
                self.model_configs.append((config_path, checkpoint_path))
                print(f"  模型 {i + 1} 加载成功")
            except Exception as e:
                print(f"  模型 {i + 1} 加载失败: {e}")
                raise e

        self.classes = [
            'other',  # 0: 其它
            'phone_use',  # 1: 打电话
            'smoking',  # 2: 抽烟
            'yawning'  # 3: 打哈欠
        ]

        # 类别中文映射（用于更好的日志显示）
        self.class_names_cn = {
            'other': '其它',
            'phone_use': '使用手机',
            'smoking': '吸烟',
            'yawning': '打哈欠',
        }

        self.mae_interence = None
        if len(mae_model_configs) > 0:
            # 初始化mae 模型
            mae_config = MaeConfig()
            self.mae_interence = VideoMAEInference(mae_model_configs[0][0], mae_config)

        print(f"\n========== 初始化完成 ==========")
        print(f"设备: {device}")
        print(f"滑动窗口参数: clip_len={clip_len}, overlap={overlap}, stride={self.stride}")
        print(f"类别数量: {len(self.classes)}")
        print(f"类别列表: {self.classes}")
        print(f"临时文件目录: {self.temp_dir}")

    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"临时目录已清理: {self.temp_dir}")

    def extract_frames(self, video_path):
        """
        从视频中提取帧

        Returns:
            frames: 帧列表 (RGB格式)
            fps: 视频帧率
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR转RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"无法读取视频: {video_path}")

        print(f"\n视频信息: {Path(video_path).name}")
        print(f"  分辨率: {width}x{height}")
        print(f"  总帧数: {len(frames)}")
        print(f"  帧率: {fps:.2f} fps")
        print(f"  时长: {len(frames) / fps:.2f} 秒")

        return frames, fps

    def inference_without_tempfile(self, model, window_frames, model_name):
        """
        使用内存中的帧数据进行推理
        """
        try:
            # 方法1：尝试使用帧列表直接推理（如果API支持）
            if hasattr(model, 'inference_frames'):
                result = model.inference_frames(window_frames)
                print(f"     {model_name}: 使用内存推理")
            else:
                # 方法2：使用临时文件（回退方案）
                temp_path = os.path.join(self.temp_dir, f"temp_{model_name}_{time.time()}.mp4")

                # 写入视频
                h, w = window_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, 15, (w, h))

                for frame in window_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()

                # 推理
                result = inference_recognizer(model, temp_path)

                # 删除临时文件
                os.remove(temp_path)

            return result
        except Exception as e:
            print(f"     {model_name} 推理出错: {e}，回退到临时文件方法")
            # 回退到临时文件方法
            temp_path = os.path.join(self.temp_dir, f"temp_{model_name}_{time.time()}.mp4")

            h, w = window_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 15, (w, h))

            for frame in window_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()

            result = inference_recognizer(model, temp_path)
            os.remove(temp_path)

            return result

    def get_slice_prediction(self, window_frames, model_idx, model, model_name):
        """
        对单个切片进行预测，返回预测类别

        Args:
            window_frames: 切片帧列表
            model_idx: 模型索引
            model: 模型对象
            model_name: 模型名称

        Returns:
            pred_class: 预测类别
            pred_score: 预测置信度
        """
        # 使用优化后的推理方法
        result = self.inference_without_tempfile(model, window_frames, model_name)

        # 提取预测结果
        if hasattr(result, 'pred_score'):
            pred_scores = result.pred_score
            pred_label = result.pred_label if hasattr(result, 'pred_label') else pred_scores.argmax().item()

            if torch.is_tensor(pred_scores):
                pred_score = pred_scores[pred_label].item()
            else:
                pred_score = pred_scores[pred_label]
        else:
            pred_label = result.pred_label
            pred_scores = result.pred_scores
            pred_score = pred_scores[pred_label].item()

        pred_class = self.classes[pred_label]
        pred_class_cn = self.class_names_cn.get(pred_class, pred_class)

        return pred_class, pred_score, pred_class_cn

    def get_majority_vote(self, predictions):
        """
        对多个模型的预测结果进行投票

        Args:
            predictions: 列表，每个元素为 (class, score, class_cn)

        Returns:
            final_class: 投票结果类别
            final_score: 平均置信度
        """
        from collections import Counter

        # 统计投票
        class_votes = Counter([p[0] for p in predictions])

        # 获取最高票数的类别
        max_votes = max(class_votes.values())
        top_classes = [cls for cls, votes in class_votes.items() if votes == max_votes]

        if len(top_classes) == 1:
            final_class = top_classes[0]
        else:
            # 平局时选择平均置信度最高的
            avg_scores = {}
            for cls in top_classes:
                scores = [p[1] for p in predictions if p[0] == cls]
                avg_scores[cls] = np.mean(scores)
            final_class = max(avg_scores, key=avg_scores.get)

        # 计算平均置信度
        scores_for_final = [p[1] for p in predictions if p[0] == final_class]
        final_score = np.mean(scores_for_final)

        return final_class, final_score

    def append_slice_result_to_file(self, video_name, start_frame, end_frame, pred_class, output_dir):
        """
        将切片结果追加到对应类别的文件中

        Args:
            video_name: 视频文件名
            start_frame: 起始帧索引
            end_frame: 结束帧索引
            pred_class: 预测类别
            output_dir: 输出目录
        """
        # 构建输出文件路径: output_dir/类别.txt
        output_file = os.path.join(output_dir, f"{pred_class}.txt")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 追加一行: 视频名 起始帧 结束帧
        line = f"{video_name} {start_frame} {end_frame}"

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

        print(f"    切片结果已写入: {output_file} -> {line}")

    def infer_video_slices(self, frames):

        # 计算窗口数量
        start_idx = 0
        window_indices = []
        while start_idx + self.clip_len <= len(frames):
            window_indices.append((start_idx, start_idx + self.clip_len))
            start_idx += self.stride

        print(f"\n总共 {len(window_indices)} 个切片")

        # 模型列表
        models_to_use = [(idx, model, name) for idx, (model, name) in enumerate(zip(self.models, self.model_names))]

        ensemble_predictions = []
        # 对每个切片进行处理
        for window_id, (start, end) in enumerate(tqdm(window_indices, desc="处理切片")):
            print(f"\n  ----- 切片 {window_id + 1}/{len(window_indices)} [帧 {start}-{end}] -----")

            # 提取当前窗口的帧
            window_frames = frames[start:end]

            # 对所有模型进行推理
            predictions = []  # 存储 (class, score, class_cn)
            for model_idx, model, model_name in models_to_use:
                pred_class, pred_score, pred_class_cn = self.get_slice_prediction(
                    window_frames, model_idx, model, model_name
                )
                predictions.append((pred_class, pred_score, pred_class_cn))
                print(f"slow    {model_name}: {pred_class} ({pred_class_cn}) - 置信度: {pred_score:.4f}")

            if self.mae_interence is not None:
                mae_result = self.mae_interence.predict(window_frames)
                pred_class = mae_result['class_name']
                pred_score = mae_result['confidence']
                pred_class_cn = self.class_names_cn.get(pred_class, pred_class)

                predictions.append((pred_class, pred_score, pred_class_cn))
                print(f"mae    {pred_class} ({pred_class_cn}) - 置信度: {pred_score:.4f}")

            # 投票决定最终类别
            final_class, final_score = self.get_majority_vote(predictions)
            final_class_cn = self.class_names_cn.get(final_class, final_class)
            print(f"  窗口{window_id + 1}  投票结果: {final_class} ({final_class_cn}) - 置信度: {final_score:.4f}")

            ensemble_predictions.append({
                'class': final_class,
                'score': final_score
            })

        final_class, final_score = self.stage1_decision_ensemble(ensemble_predictions)
        return final_class, final_score

    def stage1_decision_ensemble(self, ensemble_predictions):
        print(f"\n========== 开始最终决策 ==========")

        if len(ensemble_predictions) == 0:
            print("  没有窗口预测结果")
            return None, 0.0

        # 统计各类别出现次数（使用集成后的类别）
        class_counts = {}
        class_scores = {}

        for pred in ensemble_predictions:
            cls = pred['class']
            score = pred['score']

            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in class_scores or score > class_scores[cls]:
                class_scores[cls] = score

        total_windows = len(ensemble_predictions)

        print(f"\n各窗口集成结果统计:")
        for i, pred in enumerate(ensemble_predictions):
            cls_cn = self.class_names_cn.get(pred['class'], pred['class'])
            print(f"  窗口 {i + 1}: {pred['class']} ({cls_cn}) - 置信度: {pred['score']:.4f}")

        print(f"\n类别统计:")
        for cls, count in class_counts.items():
            cls_cn = self.class_names_cn.get(cls, cls)
            print(f"  {cls} ({cls_cn}): {count}/{total_windows} 窗口, 最高置信度: {class_scores[cls]:.4f}")

        # 获取非other类别的统计
        non_other_classes = ['phone_use', 'smoking', 'yawning']
        non_other_counts = {cls: count for cls, count in class_counts.items() if cls in non_other_classes}

        # 条件1：如果全是other，则输出other
        if class_counts.get('other', 0) == total_windows:
            print(f"\n决策结果(条件1): 所有窗口都是other，输出 other")
            return 'other', class_scores.get('other', 0.0)

        # 条件2：如果有phone_use, smoking, yawning中的类别
        if non_other_counts:
            # 找出出现次数最多的非other类别
            max_count = max(non_other_counts.values())
            most_freq_non_other = [cls for cls, count in non_other_counts.items() if count == max_count]

            if len(most_freq_non_other) == 1:
                # 只有一个类别出现次数最多，直接返回
                pred_cls = most_freq_non_other[0]
                cls_cn = self.class_names_cn.get(pred_cls, pred_cls)
                print(f"\n决策结果(条件2): {pred_cls} ({cls_cn})出现次数最多({max_count}/{total_windows})，输出该类别")
                return pred_cls, class_scores[pred_cls]
            else:
                # 出现次数一样多，取最高置信度的
                max_score = -1
                pred_cls = None
                for cls in most_freq_non_other:
                    score = class_scores.get(cls, 0)
                    if score > max_score:
                        max_score = score
                        pred_cls = cls

                cls_cn = self.class_names_cn.get(pred_cls, pred_cls)
                print(
                    f"\n决策结果(条件2): {most_freq_non_other}出现次数相同({max_count}/{total_windows})，取置信度最高的 {pred_cls} ({cls_cn})")
                return pred_cls, max_score

        # 条件3：如果没有phone_use, smoking, yawning，但有other
        print(f"\n决策结果(条件3): 没有检测到phone_use/smoking/yawning，输出 other")
        return 'other', class_scores.get('other', 0.0)
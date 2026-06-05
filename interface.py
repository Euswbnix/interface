import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import tempfile
import warnings
import logging
import sys

import time
from datetime import datetime

# 兼容 PyTorch 2.6+：torch.load 默认 weights_only=True，会导致 mmengine 加载
# 含 HistoryBuffer 的 checkpoint 失败。这里的权重均为本地可信训练产物，
# 因此将默认值恢复为 weights_only=False。
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# MMAction2 相关导入
from mmaction.apis import inference_recognizer, init_recognizer
from config import InterfaceConfig
from first_step.first_interface import FirstSliceInference
from second_step.second_interface import SecondSliceInference


class LoggerSetup:
    """日志设置类，将输出同时写入文件和控制台"""

    @staticmethod
    def setup_logger(log_dir='./logs', log_level=logging.INFO):
        """设置日志记录器"""
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'inference_{timestamp}.log'

        # 创建日志记录器
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # 清除已有的处理器
        logger.handlers.clear()

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)

        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 重定向print到logging
        class PrintLogger:
            def __init__(self, logger, level=logging.INFO):
                self.logger = logger
                self.level = level

            def write(self, message):
                if message.strip():  # 忽略空消息
                    self.logger.log(self.level, message.strip())

            def flush(self):
                pass

        # 重定向标准输出
        sys.stdout = PrintLogger(logger, logging.INFO)
        sys.stderr = PrintLogger(logger, logging.ERROR)

        print(f"日志文件保存在: {log_file}")
        return logger, log_file


class SliceInference:
    """视频切片行为识别推理类 - 每个切片单独输出结果"""

    def __init__(self, logger=None):
        config = InterfaceConfig()
        self.logger = logger or logging.getLogger()
        self.clip_len = config.clip_len
        self.overlap = config.overlap
        self.stride = self.clip_len - self.overlap
        self.first_interface = FirstSliceInference(slow_model_configs=config.first_slow_model_configs,
                                                   mae_model_configs=config.first_mae_model_configs,
                                                   device=config.device,
                                                   clip_len=config.clip_len,
                                                   overlap=config.overlap,
                                                   logger=self.logger)

        self.second_interface = SecondSliceInference(slow_model_configs=config.second_slow_model_configs,
                                                     mae_model_configs=config.second_mae_model_configs,
                                                     device=config.device,
                                                     clip_len=config.clip_len,
                                                     overlap=config.overlap,
                                                     logger=self.logger)

    def __del__(self):
        """清理临时文件"""

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

    def append_slice_result_to_file(self, video_name, confidence, pred_class, output_dir):
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
        line = f"{video_name} {confidence}"

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

        print(f"    切片结果已写入: {output_file} -> {line}")

    def infer_folder_slices(self, input_dir, output_dir=None,
                            video_extensions=['.mp4']):
        """
        遍历文件夹进行切片推理
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

        # 如果未指定输出目录，则在输入目录下创建结果文件夹
        if output_dir is None:
            output_dir = input_path / 'slice_results'

        # 收集所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))

        # 去重并排序
        video_files = list(set(video_files))
        video_files.sort()

        print(f"\n{'=' * 60}")
        print(f"开始批量切片推理")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"找到 {len(video_files)} 个视频文件")
        print(f"{'=' * 60}\n")

        for i, video_file in enumerate(video_files, 1):
            print(f"\n--- 处理视频 {i}/{len(video_files)} ---")
            """
            对单个视频进行切片推理，每个切片单独输出到对应类别的文件

            Args:
                video_path: 视频路径
            """
            video_name = Path(str(video_file)).name
            print(f"\n{'=' * 60}")
            print(f"开始推理视频: {video_name}")
            print(f"{'=' * 60}")

            # 提取帧
            try:
                start_time_frame = time.time()
                frames, fps = self.extract_frames(str(video_file))
                end_time_frame = time.time()
                frame_time = end_time_frame - start_time_frame
                print(f"抽帧耗时: {frame_time:.4f} 秒")
            except Exception as e:
                print(f"  提取帧失败: {e}")
                return {'video': video_name, 'error': str(e)}

            if len(frames) < self.clip_len:
                print(f"  警告: 视频帧数({len(frames)})小于窗口长度({self.clip_len})")
                return {'video': video_name, 'error': f'视频太短，只有{len(frames)}帧'}

            # 计算窗口数量
            start_idx = 0
            window_indices = []
            while start_idx + self.clip_len <= len(frames):
                window_indices.append((start_idx, start_idx + self.clip_len))
                start_idx += self.stride

            print(f"\n总共 {len(window_indices)} 个切片")

            try:
                print(f"step1开始")
                start_time_analy = time.time()
                first_final_class, first_final_score = self.first_interface.infer_video_slices(frames)
                end_time_analy = time.time()
                execution_time = end_time_analy - start_time_analy
                print(
                    f"step1结束 {video_file.name} 耗时: {execution_time:.4f} 秒  最终结果: {first_final_class}, 置信度:{first_final_score}")

                if first_final_class != 'other':
                    # 直接输出
                    print(f"统计最终结果: {first_final_class}, 置信度:{first_final_score} {video_file.name}")
                    self.append_slice_result_to_file(video_name, first_final_score, first_final_class, output_dir)
                    continue

                print(f"step2开始")
                start_time_2 = time.time()
                second_final_class, second_final_score = self.second_interface.infer_video_slices(frames)
                end_time_2 = time.time()
                execution_2 = end_time_2 - start_time_2
                print(
                    f"step2结束 {video_file.name} 耗时: {execution_2:.4f} 秒  最终结果: {second_final_class}, 置信度:{second_final_score}")
                print(f"统计最终结果: {second_final_class}, 置信度:{second_final_score} {video_file.name}")

                # 将切片结果写入对应类别的文件
                self.append_slice_result_to_file(video_name, second_final_score, second_final_class, output_dir)

            except Exception as e:
                print(f"视频 {video_file.name} 处理失败: {e}")


def check_first_pth_exist():
    print("\n检查step1模型路径...")
    interfaceConfig = InterfaceConfig()

    slow_model_configs = interfaceConfig.first_slow_model_configs
    mae_model_configs = interfaceConfig.first_mae_model_configs

    if len(slow_model_configs) == 0 and len(mae_model_configs) == 0:
        return False

    # 检查文件是否存在
    slow_missing_files = []
    slow_valid_configs = []

    for config_path, checkpoint_path, model_name in slow_model_configs:
        config_exists = Path(config_path).exists()
        checkpoint_exists = Path(checkpoint_path).exists()

        if not config_exists:
            slow_missing_files.append(f"配置文件 {model_name}: {config_path}")
        if not checkpoint_exists:
            slow_missing_files.append(f"权重文件 {model_name}: {checkpoint_path}")

        if config_exists and checkpoint_exists:
            slow_valid_configs.append((config_path, checkpoint_path, model_name))
        else:
            print(f"跳过模型 {model_name} (文件不存在)")

    if slow_missing_files:
        print("\n警告: 以下文件不存在:")
        for msg in slow_missing_files:
            print(f"  {msg}")
        return False

    print(f"\n有效slow模型数量: {len(slow_valid_configs)}")

    mae_missing_files = []
    mae_valid_configs = []
    for checkpoint_path, model_name in mae_model_configs:
        checkpoint_exists = Path(checkpoint_path).exists()

        if not checkpoint_exists:
            mae_missing_files.append(f"权重文件 {model_name}: {checkpoint_path}")

        if checkpoint_exists:
            mae_valid_configs.append((checkpoint_path, model_name))
        else:
            print(f"跳过模型 {model_name} (文件不存在)")

    if mae_missing_files:
        print("\n警告: 以下文件不存在:")
        for msg in mae_missing_files:
            print(f"  {msg}")
        return False

    print(f"\n有效mae模型数量: {len(mae_valid_configs)}")

    return True


def check_second_pth_exist():
    print("\n检查step2模型路径...")
    interfaceConfig = InterfaceConfig()

    slow_model_configs = interfaceConfig.second_slow_model_configs
    mae_model_configs = interfaceConfig.second_mae_model_configs

    if len(slow_model_configs) == 0 and len(mae_model_configs) == 0:
        return False

    # 检查文件是否存在
    slow_missing_files = []
    slow_valid_configs = []

    for config_path, checkpoint_path, model_name in slow_model_configs:
        config_exists = Path(config_path).exists()
        checkpoint_exists = Path(checkpoint_path).exists()

        if not config_exists:
            slow_missing_files.append(f"配置文件 {model_name}: {config_path}")
        if not checkpoint_exists:
            slow_missing_files.append(f"权重文件 {model_name}: {checkpoint_path}")

        if config_exists and checkpoint_exists:
            slow_valid_configs.append((config_path, checkpoint_path, model_name))
        else:
            print(f"跳过模型 {model_name} (文件不存在)")

    if slow_missing_files:
        print("\n警告: 以下文件不存在:")
        for msg in slow_missing_files:
            print(f"  {msg}")
        return False

    print(f"\n有效slow模型数量: {len(slow_valid_configs)}")

    mae_missing_files = []
    mae_valid_configs = []
    for checkpoint_path, model_name in mae_model_configs:
        checkpoint_exists = Path(checkpoint_path).exists()

        if not checkpoint_exists:
            mae_missing_files.append(f"权重文件 {model_name}: {checkpoint_path}")

        if checkpoint_exists:
            mae_valid_configs.append((checkpoint_path, model_name))
        else:
            print(f"跳过模型 {model_name} (文件不存在)")

    if mae_missing_files:
        print("\n警告: 以下文件不存在:")
        for msg in mae_missing_files:
            print(f"  {msg}")
        return False

    print(f"\n有效mae模型数量: {len(mae_valid_configs)}")

    return True


def main():
    parser = argparse.ArgumentParser(description='视频切片行为识别推理 - 每个切片单独输出')
    parser.add_argument('--input_dir', type=str, required=True, help='输入视频文件夹路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出结果文件夹路径')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志文件保存路径')
    parser.add_argument('--no_console_log', action='store_true', help='不在控制台输出日志')
    args = parser.parse_args()

    # 设置日志
    logger, log_file = LoggerSetup.setup_logger(args.log_dir)

    # 如果不需要控制台输出，移除控制台处理器
    if args.no_console_log:
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.removeHandler(handler)

    step1_check_ret = check_first_pth_exist()
    if step1_check_ret is False:
        print("\nstep1模型路径失败")
        return

    step2_check_ret = check_second_pth_exist()
    if step2_check_ret is False:
        print("\nstep1模型路径失败")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"推理开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"日志文件: {log_file}")

    # 初始化推理器
    inferencer = SliceInference(
        logger=logger
    )

    # 执行推理
    try:
        results = inferencer.infer_folder_slices(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )

    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"推理结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件保存在: {log_file}")
    print(f"{'=' * 60}")


# python interface.py --input_dir /video/test/biyan/ --log_dir biyan_logs --output_dir biyan_out
if __name__ == '__main__':
    main()


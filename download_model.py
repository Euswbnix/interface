# download_model.py
from transformers import VideoMAEForVideoClassification

# 设置镜像源（可选）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载并保存到本地缓存
model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base')
model.save_pretrained('./videomae-base-local')

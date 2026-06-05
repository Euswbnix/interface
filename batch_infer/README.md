# 批量下载 + 推理（5.1~5.20，三张表）

清单 `manifest_5.1-5.20.csv`：113,073 个唯一视频（已按风险时间升序、按 hash 去重），约 339 GB。
文件名 = `<hash>.mp4`，`hash` 即报警ID末尾 32 位，用于把结果映射回 Excel。

## 流程总览
```
本地: build_manifest 已生成 manifest_5.1-5.20.csv
  │ (随仓库一起到服务器)
服务器(5090):
  1) download.py   清单 -> /data/videos/<hash>.mp4   (并发/断点续传)
  2) run_infer.py  逐个推理 -> results_5.1-5.20.jsonl (模型只加载一次/断点续跑)
  │ (把 jsonl 拷回本地)
本地:
  3) join_results.py  jsonl + xlsx -> 对比CSV（含重复引用的行）
```

## 1. 下载（服务器）
```bash
cd interface
python batch_infer/download.py \
    --manifest batch_infer/manifest_5.1-5.20.csv \
    --out-dir  /data/videos \
    --workers  16
```
- 中断后**重跑同一条命令**即可续传（已下完的自动跳过）。
- 失败项写入 `failed.csv`，单独重跑：`--manifest failed.csv`。
- 磁盘需 ~340 GB。磁盘紧张就用下面的「边下边推」。

## 2. 推理（服务器）
```bash
cd interface
python batch_infer/run_infer.py \
    --manifest  batch_infer/manifest_5.1-5.20.csv \
    --video-dir /data/videos \
    --out       batch_infer/results_5.1-5.20.jsonl
```
- 模型只加载一次，按清单顺序跑；中断后重跑自动从断点继续。
- 还没下到的视频会被跳过（`missing`），下完再跑会补上。
- 后台长跑建议：`nohup python batch_infer/run_infer.py ... > infer.log 2>&1 &`

### 边下边推、不留存（磁盘紧张时）
推理加 `--delete`：每个视频推完即删。两条命令同时开着，下载在前、推理在后追，磁盘峰值只占缓冲。
```bash
python batch_infer/run_infer.py ... --delete
```

推理输出极简：模型内部逐切片日志全部静音，只剩一个总体进度条
`推理 |████| 1234/113073`。要看曲线加 `--swanlab`（需 `pip install swanlab`）。

默认每处理 500 条会打印一次近期预测样本 + 类别累计分布（用 `tqdm.write` 不打断进度条）：
- `--print-every N` 改间隔（0 关闭），`--print-samples K` 改展示条数。
```
[20:23:52] 已处理 500 (ok=500 err=0 miss=0) 3.21/s
  近 5 条预测: b63c1dd8 -> yawning(0.979) s2 | 72f78920 -> phone_use(0.415) s2 | ...
  类别累计分布: phone_use=97 smoking=96 biyan=88 normal=80 zhuyili_no_focus=72 yawning=67
```

## 3. 回填对比（本地）
```bash
python batch_infer/join_results.py \
    --xlsx    ../高风险事件样本.xlsx \
    --results results_5.1-5.20.jsonl \
    --out     results_joined_5.1-5.20.csv
```
输出含：报警ID、风险开始时间、DMS模型结果、标注结果、模型预测类别/置信度/阶段。

## 4. 混淆矩阵 + 二分类统计（本地，需 numpy/matplotlib）
```bash
python batch_infer/analyze.py \
    --xlsx    ../高风险事件样本.xlsx \
    --results results_5.1-5.20.jsonl \
    --outdir  analysis_5.1-5.20
```
产出：
- `report.md` —— 取值统计、TP/FP/TN/FN 在本场景的含义、两组二分类指标（全量 + 仅覆盖行为）
- `cm_multiclass_newmodel.png` —— 多分类混淆矩阵（真值 × 新模型）
- `cm_binary_dms.png` / `cm_binary_newmodel_full.png` / `cm_binary_newmodel_covered.png`

二分类定义：正=真报（真实风险行为），负=误报/无视频；设备异常类排除。
DMS 是报警生成器（恒判真报，只有 TP/FP）；新模型是复核器（完整 TP/FP/TN/FN）。
正/负归类在 `analyze.py` 顶部的映射表，可随时改。

## 注意
- 视频链接 8/3 过期，三张表请在此之前下完。
- 类别：`other/phone_use(打电话)/smoking(抽烟)/yawning(打哈欠)`。
- GPU(`cuda:0`)、模型权重路径在 `../config.py`，确认权重在服务器上存在。

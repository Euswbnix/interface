#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混淆矩阵 / 二分类统计 —— 在本地运行（需 高风险事件样本.xlsx + openpyxl + numpy + matplotlib）。

三方标签：
  真值   = 标注结果（对应链接）      —— ground truth
  DMS    = DMS模型结果              —— 原模型预测（报警生成器，恒判“有风险”）
  新模型 = results_*.jsonl 的 class —— interface 新模型预测（other/phone_use/smoking/yawning）

产出：
  - report.md                       文字报告：取值统计、TP/FP/TN/FN 在本场景的含义、各项指标
  - cm_multiclass_newmodel.png      多分类混淆矩阵（真值 × 新模型，4 类空间）
  - cm_binary_dms.png               二分类：DMS vs 真值
  - cm_binary_newmodel_full.png     二分类：新模型 vs 真值（全量）
  - cm_binary_newmodel_covered.png  二分类：新模型 vs 真值（仅覆盖行为）

用法：
  python analyze.py \
      --xlsx    ../../高风险事件样本.xlsx \
      --results results_5.1-5.20.jsonl \
      --outdir  analysis_5.1-5.20
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import openpyxl

SHEETS = ["5.1~5.6", "5.7~5.13", "5.14~5.20"]

# ===== 标签映射（可按需改）=====================================================
# 真值（标注）→ 二分类
TRUTH_POSITIVE = {"频繁低头", "玩手机", "打哈欠", "抽烟", "注意力分散", "打电话", "真实闭眼"}  # 真报
TRUTH_NEGATIVE = {"误报", "无视频"}                                                       # 误报/无效
TRUTH_EXCLUDE = {"摄像头遮挡", "摄像头扭转", "脱离监控", "驾驶员异常"}                       # 设备异常，二分类中排除

# “仅覆盖行为”范围：新模型能识别的行为 + 误报负类
COVERED_POSITIVE = {"玩手机", "打哈欠", "抽烟", "打电话"}
COVERED_NEGATIVE = {"误报", "无视频"}

# 新模型类别 → 二分类（判“真报”还是“误报”）
MODEL_POS = {"phone_use", "smoking", "yawning"}
MODEL_NEG = {"other"}

# 多分类公共 4 类空间：真值/新模型都映射到这里
def truth_to_4cls(t):
    if t in ("玩手机", "打电话"):
        return "phone"
    if t == "打哈欠":
        return "yawning"
    if t == "抽烟":
        return "smoking"
    return "other"

def model_to_4cls(c):
    return {"phone_use": "phone", "yawning": "yawning", "smoking": "smoking"}.get(c, "other")

FOUR = ["phone", "smoking", "yawning", "other"]
# ============================================================================


def load_results(path):
    res = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            res[r["hash"]] = r.get("class")  # None 表示推理出错
    return res


def load_rows(xlsx):
    """返回每个有结果行: dict(truth, dms, hash)。"""
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    rows = []
    for sn in SHEETS:
        ws = wb[sn]
        h = list(next(ws.iter_rows(values_only=True)))
        iu = h.index("唤醒视频链接")
        idms = h.index("DMS模型结果")
        ilab = h.index("标注结果（对应链接）")
        for r in ws.iter_rows(min_row=2, values_only=True):
            u = r[iu]
            if not u or not str(u).strip():
                continue
            hsh = str(u).rsplit("/", 1)[-1].replace(".mp4", "").split("_")[-1]
            rows.append({
                "hash": hsh,
                "truth": (r[ilab] or "").strip(),
                "dms": (r[idms] or "").strip(),
            })
    return rows


def binary_counts(actual_pred_pairs):
    """pairs: list of (actual P/N bool, pred P/N bool). 返回 TP,FP,TN,FN。"""
    TP = FP = TN = FN = 0
    for a, p in actual_pred_pairs:
        if a and p:
            TP += 1
        elif a and not p:
            FN += 1
        elif (not a) and p:
            FP += 1
        else:
            TN += 1
    return TP, FP, TN, FN


def metrics(TP, FP, TN, FN):
    n = TP + FP + TN + FN
    acc = (TP + TN) / n if n else 0
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    spec = TN / (TN + FP) if (TN + FP) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    fpr = FP / (TN + FP) if (TN + FP) else float("nan")  # 误报率（把误报当真报）
    fnr = FN / (TP + FN) if (TP + FN) else float("nan")  # 漏报率（漏掉真报）
    return dict(n=n, acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, fpr=fpr, fnr=fnr)


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) and x == x else "—"


def plot_cm(mat, xlabels, ylabels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(xlabels), 1.5 + 1.1 * len(ylabels)))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(xlabels)), labels=xlabels)
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (truth)")
    ax.set_title(title)
    vmax = mat.max() if mat.size else 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = load_results(args.results)
    rows = load_rows(args.xlsx)
    # 只保留有新模型结果且推理成功的行
    rows = [r for r in rows if r["hash"] in results and results[r["hash"]] is not None]
    for r in rows:
        r["model"] = results[r["hash"]]
    print(f"参与统计行数（有效新模型结果）: {len(rows)}")

    truth_cnt = Counter(r["truth"] for r in rows)
    dms_cnt = Counter(r["dms"] for r in rows)
    model_cnt = Counter(r["model"] for r in rows)

    # ---- 多分类混淆矩阵：真值(4类) × 新模型(4类) ----
    idx = {c: i for i, c in enumerate(FOUR)}
    mc = np.zeros((4, 4), dtype=int)
    for r in rows:
        mc[idx[truth_to_4cls(r["truth"])], idx[model_to_4cls(r["model"])]] += 1
    plot_cm(mc, FOUR, FOUR, "Multiclass: truth x new-model",
            os.path.join(args.outdir, "cm_multiclass_newmodel.png"))

    # ---- 二分类 ----
    def actual_of(truth, pos, neg):
        if truth in pos:
            return True
        if truth in neg:
            return False
        return None  # 排除

    # DMS：恒判“真报”(positive)；新模型：MODEL_POS→真报
    def make_pairs(model_kind, pos, neg):
        pairs = []
        for r in rows:
            a = actual_of(r["truth"], pos, neg)
            if a is None:
                continue
            if model_kind == "dms":
                p = True  # DMS 永远报警 → 判正
            else:
                p = r["model"] in MODEL_POS  # other→False
            pairs.append((a, p))
        return pairs

    blocks = []  # (名称, TP,FP,TN,FN, metrics)
    configs = [
        ("DMS vs 真值（全量）", "DMS vs truth (all)", "dms", TRUTH_POSITIVE, TRUTH_NEGATIVE, "cm_binary_dms.png"),
        ("新模型 vs 真值（全量）", "new-model vs truth (all)", "new", TRUTH_POSITIVE, TRUTH_NEGATIVE, "cm_binary_newmodel_full.png"),
        ("新模型 vs 真值（仅覆盖行为）", "new-model vs truth (covered only)", "new", COVERED_POSITIVE, COVERED_NEGATIVE, "cm_binary_newmodel_covered.png"),
    ]
    for name, ascii_title, kind, pos, neg, png in configs:
        TP, FP, TN, FN = binary_counts(make_pairs(kind, pos, neg))
        m = metrics(TP, FP, TN, FN)
        blocks.append((name, TP, FP, TN, FN, m))
        plot_cm([[TP, FN], [FP, TN]], ["Pred Pos", "Pred Neg"],
                ["Real(P)", "False(N)"],
                ascii_title, os.path.join(args.outdir, png))

    # ---- 写报告 ----
    rp = os.path.join(args.outdir, "report.md")
    with open(rp, "w", encoding="utf-8") as f:
        w = f.write
        w("# 混淆矩阵与二分类统计报告\n\n")
        w(f"参与统计行数（有有效新模型结果）：**{len(rows)}**\n\n")

        w("## 一、取值统计\n\n")
        for title, cnt in [("真值（标注结果）", truth_cnt), ("DMS模型结果", dms_cnt), ("新模型预测", model_cnt)]:
            w(f"**{title}**\n\n")
            for k, v in cnt.most_common():
                w(f"- {k or '∅'}: {v}\n")
            w("\n")

        w("## 二、二分类正/负定义\n\n")
        w("- **正类 = 真报（真实风险行为）**：" + "、".join(sorted(TRUTH_POSITIVE)) + "\n")
        w("- **负类 = 误报/无效**：" + "、".join(sorted(TRUTH_NEGATIVE)) + "\n")
        w("- **排除（设备异常，不计入二分类）**：" + "、".join(sorted(TRUTH_EXCLUDE)) + "\n\n")

        w("## 三、TP/FP/TN/FN 在本场景代表什么\n\n")
        w("以「该报警是否为真实风险」作判断：\n\n")
        w("- **TP（真阳）**：真值是真实风险行为，模型也判为真报 → 正确确认风险。\n")
        w("- **FP（假阳）**：真值是误报，模型却判为真报 → 把误报当成真风险，**误报未被过滤**。\n")
        w("- **TN（真阴）**：真值是误报，模型也判为误报(other) → **正确识别并过滤掉误报**。\n")
        w("- **FN（假阴）**：真值是真实风险，模型却判为误报(other) → **漏掉真实风险（漏报，后果最严重）**。\n\n")
        w("> 说明：DMS 是“报警生成器”，本样本里每一行都是 DMS 已报的警，所以 DMS 永远判“真报”"
          "（恒为 pred 真报）。因此 DMS 的混淆矩阵只有 TP / FP，没有 TN / FN——它衡量的是"
          "**DMS 报警的精确率/误报率**。新模型作为“二次复核器”能输出 other(=误报)，"
          "因此具备完整的 TP/FP/TN/FN，衡量它**能否在 DMS 的报警里挑出误报(TN)、代价是漏掉多少真报(FN)**。\n\n")
        w("> 全量 vs 仅覆盖行为：新模型只识别 手机/抽烟/打哈欠 三类，对真值里的 频繁低头/闭眼/注意力分散 "
          "会一律判 other。**全量版**把这些真报算作新模型的 FN（反映真实过滤现状，漏报偏高）；"
          "**仅覆盖行为版**只在新模型设计覆盖的行为+误报上评估（公平反映模型本身能力）。\n\n")

        w("## 四、两组二分类混淆矩阵与指标\n\n")
        for name, TP, FP, TN, FN, m in blocks:
            w(f"### {name}\n\n")
            w("| 真值\\预测 | pred 真报 | pred 误报 |\n|---|---|---|\n")
            w(f"| **真报(P)** | TP={TP} | FN={FN} |\n")
            w(f"| **误报(N)** | FP={FP} | TN={TN} |\n\n")
            w(f"- 样本数 n={m['n']}；准确率 acc={fmt(m['acc'])}\n")
            w(f"- 精确率 precision={fmt(m['prec'])}（判为真报中真的是真报的比例）\n")
            w(f"- 召回率 recall={fmt(m['rec'])}（真报被判出的比例）\n")
            w(f"- 特异度 specificity={fmt(m['spec'])}（误报被正确过滤的比例）\n")
            w(f"- F1={fmt(m['f1'])}\n")
            w(f"- 误报率 FPR={fmt(m['fpr'])}（误报被当成真报）；漏报率 FNR={fmt(m['fnr'])}（真报被漏掉）\n\n")

        w("## 五、图表\n\n")
        w("- `cm_multiclass_newmodel.png` 多分类混淆矩阵（真值 × 新模型，phone/smoking/yawning/other）\n")
        w("- `cm_binary_dms.png` / `cm_binary_newmodel_full.png` / `cm_binary_newmodel_covered.png` 三张二分类混淆矩阵\n")

    print(f"报告已写出: {rp}")
    print("图表:", os.listdir(args.outdir))


if __name__ == "__main__":
    main()

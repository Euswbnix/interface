#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并所有分片结果（含之前单进程跑的结果），按 hash 去重（后写覆盖），输出一个总文件。

用法（在 interface 目录下）：
  python batch_infer/merge_results.py \
      --glob "batch_infer/results_5.1-5.20*.jsonl" \
      --out  batch_infer/results_5.1-5.20.merged.jsonl

之后用 results_5.1-5.20.merged.jsonl 跑 analyze.py / join_results.py。
"""
import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="batch_infer/results_5.1-5.20*.jsonl",
                    help="匹配所有分片+单进程结果文件")
    ap.add_argument("--out", default="batch_infer/results_5.1-5.20.merged.jsonl")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(args.glob) if f != args.out)
    print(f"合并 {len(files)} 个文件: {files}")

    merged = {}          # hash -> record（后出现的覆盖）
    n_lines = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa
                    continue
                n_lines += 1
                merged[r["hash"]] = r

    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_err = sum(1 for r in merged.values() if r.get("class") is None)
    print(f"读入 {n_lines} 行，去重后 {len(merged)} 条唯一结果（其中推理失败 {n_err}）")
    print(f"已写出: {args.out}")


if __name__ == "__main__":
    main()

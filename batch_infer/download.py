#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载唤醒视频 —— 在服务器上运行。

特性：
  - 多线程并发下载
  - 断点续跑：已存在且完整的 .mp4 自动跳过；中断后重跑即可继续
  - 原子落盘：先写 .part，校验大小后再 rename，绝不会留下半截 mp4
  - 失败重试 + 失败清单（failed.csv），可单独重跑失败项
  - 按 manifest 顺序（已按风险时间升序）下载

用法：
  python download.py \
      --manifest manifest_5.1-5.20.csv \
      --out-dir  /data/videos \
      --workers  16

只重跑失败项：
  python download.py --manifest failed.csv --out-dir /data/videos --workers 16
"""
import argparse
import csv
import os
import sys
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "Mozilla/5.0 (batch-downloader)"

_print_lock = threading.Lock()
_counter_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0}


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def load_manifest(path):
    """返回 [(hash, url), ...]。兼容主 manifest 和 failed.csv（都含 hash,url 两列）。"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            h = (row.get("hash") or "").strip()
            u = (row.get("url") or "").strip()
            if h and u:
                rows.append((h, u))
    return rows


def download_one(h, url, out_dir, retries, timeout):
    """下载单个视频。返回 ('ok'|'skip'|'fail', hash, url, err)。"""
    final = os.path.join(out_dir, h + ".mp4")
    if os.path.exists(final) and os.path.getsize(final) > 0:
        with _counter_lock:
            _stats["skip"] += 1
        return ("skip", h, url, "")

    tmp = final + ".part"
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                clen = resp.getheader("Content-Length")
                clen = int(clen) if clen and clen.isdigit() else None
                with open(tmp, "wb") as out:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
            size = os.path.getsize(tmp)
            if size == 0 or (clen is not None and size != clen):
                raise IOError(f"size mismatch got={size} expect={clen}")
            os.replace(tmp, final)  # 原子
            with _counter_lock:
                _stats["ok"] += 1
            return ("ok", h, url, "")
        except Exception as e:  # noqa
            last_err = f"{type(e).__name__}: {e}"
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
    with _counter_lock:
        _stats["fail"] += 1
    return ("fail", h, url, last_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--failed-out", default="failed.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    items = load_manifest(args.manifest)
    total = len(items)
    log(f"清单共 {total} 条，输出目录 {args.out_dir}，并发 {args.workers}")

    t0 = time.time()
    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, h, u, args.out_dir, args.retries, args.timeout): h
                for h, u in items}
        for fut in as_completed(futs):
            status, h, url, err = fut.result()
            done += 1
            if status == "fail":
                failures.append((h, url, err))
            if done % 500 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                log(f"[{done}/{total}] ok={_stats['ok']} skip={_stats['skip']} "
                    f"fail={_stats['fail']} {rate:.1f}/s ETA {eta/3600:.2f}h")

    if failures:
        with open(args.failed_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hash", "url", "err"])
            w.writerows(failures)
        log(f"失败 {len(failures)} 条，已写入 {args.failed_out}（可直接当 manifest 重跑）")
    else:
        log("全部成功，无失败项。")

    log(f"完成：ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']} "
        f"用时 {(time.time()-t0)/3600:.2f}h")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

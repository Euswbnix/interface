#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理 —— 在服务器(5090)上运行。

复用 interface.py 的 SliceInference（模型只在启动时加载一次），
按 manifest 顺序逐个处理已下载的视频，结果以 jsonl 追加写出，可断点续跑。

输出极简：模型内部的逐切片打印全部静音，只保留一个总体 tqdm 进度条；
也可 --swanlab 把进度/计数上报到 SwanLab 做可视化。

用法：
  cd interface
  python batch_infer/run_infer.py \
      --manifest  batch_infer/manifest_5.1-5.20.csv \
      --video-dir /data/videos \
      --out       batch_infer/results_5.1-5.20.jsonl

可选：--delete 推完即删；--swanlab 上报 SwanLab。
"""
import argparse
import atexit
import csv
import json
import logging
import os
import signal
import sys
import time
import contextlib

# 多进程分片并行时，必须把每个进程限制成单线程，否则每进程默认按全部核数起线程，
# N 个进程会互相抢 CPU 反而更慢。这些环境变量须在 numpy/torch 导入前设置。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# 让 config / first_step / second_step / interface 可导入（仓库根目录）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm

# 静音 mmengine / mmaction / 第三方库的 logging 噪声
for name in ("", "mmengine", "mmaction", "mmcv"):
    logging.getLogger(name).setLevel(logging.ERROR)

from interface import SliceInference  # noqa: E402

# 进一步把 torch / opencv 也锁成单线程（分片并行下每片只吃 ~1 核）
import torch  # noqa: E402
try:
    torch.set_num_threads(1)
except Exception:  # noqa
    pass
try:
    import cv2  # noqa: E402
    cv2.setNumThreads(1)
except Exception:  # noqa
    pass

_DEVNULL = open(os.devnull, "w")


@contextlib.contextmanager
def quiet():
    """临时把 stdout/stderr 重定向到 /dev/null，屏蔽模型内部的 print 刷屏。"""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _DEVNULL, _DEVNULL
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def load_order(manifest):
    order = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = (row.get("hash") or "").strip()
            if h:
                order.append(h)
    return order


def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["hash"])
                except Exception:  # noqa
                    pass
    return done


def infer_one(infer, video_path):
    """两阶段推理，返回 (final_class, final_score, stage)。"""
    frames, _ = infer.extract_frames(video_path)
    if len(frames) < infer.clip_len:
        raise ValueError(f"视频太短: {len(frames)} 帧 < clip_len {infer.clip_len}")
    c1, s1 = infer.first_interface.infer_video_slices(frames)
    if c1 != "other":
        return c1, float(s1), 1
    c2, s2 = infer.second_interface.infer_video_slices(frames)
    return c2, float(s2), 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delete", action="store_true", help="推理后删除视频文件以节省磁盘")
    ap.add_argument("--swanlab", action="store_true", help="上报进度到 SwanLab 可视化")
    ap.add_argument("--swanlab-project", default="dms-batch-infer")
    ap.add_argument("--print-every", type=int, default=500,
                    help="每处理 N 条打印一次近期预测样本+类别分布(0=关闭)")
    ap.add_argument("--print-samples", type=int, default=5,
                    help="每次打印展示的近期样本条数")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="把清单切成 N 片并行（配合 --shard-id 多进程各跑一片）")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="本进程负责第几片（0..N-1）")
    args = ap.parse_args()

    order = load_order(args.manifest)
    # ---- 分片：每个进程只取属于自己的那一片，并写独立的结果文件 ----
    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards, "shard-id 必须在 0..num_shards-1"
        order = order[args.shard_id::args.num_shards]
        base, ext = os.path.splitext(args.out)
        args.out = f"{base}.shard{args.shard_id}{ext}"
        print(f"[分片 {args.shard_id}/{args.num_shards}] 本片 {len(order)} 条，"
              f"结果 -> {args.out}", flush=True)

    done = load_done(args.out)
    todo = [h for h in order if h not in done]
    print(f"清单(本片) {len(order)} 条，已完成 {len(done)} 条，本次待处理 {len(todo)} 条", flush=True)

    run = None
    if args.swanlab:
        try:
            import swanlab
            run = swanlab.init(project=args.swanlab_project,
                               config={"total": len(order), "todo": len(todo)})
        except Exception as e:  # noqa
            print(f"[warn] SwanLab 初始化失败，跳过：{e}", flush=True)
            run = None

    with quiet():
        infer = SliceInference()

    t0 = time.time()
    n_ok = n_err = n_missing = 0
    from collections import Counter, deque
    cls_dist = Counter()                       # 预测类别累计分布
    recent = deque(maxlen=max(1, args.print_samples))  # 最近若干条预测

    # ---- 结果持久化：追加写(绝不截断已有结果) + 周期 fsync 真正落盘 ----
    out = open(args.out, "a", encoding="utf-8")

    def _durable_close():
        try:
            out.flush()
            os.fsync(out.fileno())
        except Exception:  # noqa
            pass
        try:
            out.close()
        except Exception:  # noqa
            pass

    # 进程被 kill / Ctrl-C 时也保证已写结果落盘，并告知文件位置
    def _on_signal(signum, frame):
        _durable_close()
        print(f"\n[已安全保存] 收到信号 {signum}，已完成 {n_ok+n_err} 条，"
              f"结果在 {os.path.abspath(args.out)}（重跑同命令自动续）", flush=True)
        os._exit(0)

    atexit.register(_durable_close)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _on_signal)

    try:
        for h in tqdm(todo, desc="推理", unit="vid", dynamic_ncols=True):
            path = os.path.join(args.video_dir, h + ".mp4")
            if not os.path.exists(path):
                n_missing += 1
                continue
            rec = {"hash": h}
            try:
                with quiet():
                    cls, score, stage = infer_one(infer, path)
                rec.update({"class": cls, "score": score, "stage": stage})
                n_ok += 1
                cls_dist[cls] += 1
                recent.append(f"{h[:12]} -> {cls}({score:.3f}) s{stage}")
            except Exception as e:  # noqa
                rec.update({"class": None, "score": None, "error": f"{type(e).__name__}: {e}"})
                n_err += 1
                cls_dist["<error>"] += 1
                recent.append(f"{h[:12]} -> ERROR {type(e).__name__}")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            processed = n_ok + n_err
            if processed % 200 == 0:           # 周期强制落盘，断电也不丢
                os.fsync(out.fileno())

            # ---- 周期性打印近期预测样本 + 类别分布（不打断进度条）----
            if args.print_every and processed % args.print_every == 0:
                el = time.time() - t0
                rate = processed / el if el else 0
                dist = " ".join(f"{k}={v}" for k, v in cls_dist.most_common())
                tqdm.write(f"\n[{time.strftime('%H:%M:%S')}] 已处理 {processed} "
                           f"(ok={n_ok} err={n_err} miss={n_missing}) {rate:.2f}/s")
                tqdm.write(f"  近 {len(recent)} 条预测: " + " | ".join(recent))
                tqdm.write(f"  类别累计分布: {dist}")
            if args.delete:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if run is not None and (n_ok + n_err) % 50 == 0:
                el = time.time() - t0
                run.log({"done": n_ok + n_err, "ok": n_ok, "err": n_err,
                         "missing": n_missing, "rate_per_s": (n_ok + n_err) / el if el else 0})
    finally:
        _durable_close()
        if run is not None:
            run.finish()

    # ---- 完成摘要：明确告诉你数据存在哪、共多少条 ----
    total_saved = len(load_done(args.out))
    summary = {
        "results_file": os.path.abspath(args.out),
        "total_saved": total_saved,
        "this_run_ok": n_ok, "this_run_err": n_err, "missing": n_missing,
        "manifest_total": len(order),
        "elapsed_hours": round((time.time() - t0) / 3600, 3),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    spath = os.path.splitext(args.out)[0] + "_summary.json"
    with open(spath, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"\n完成：本次 ok={n_ok} err={n_err} missing={n_missing} "
          f"用时 {summary['elapsed_hours']}h", flush=True)
    print(f"[数据已保存] 累计 {total_saved}/{len(order)} 条结果 -> "
          f"{os.path.abspath(args.out)}", flush=True)
    print(f"[摘要] {spath}", flush=True)
    if total_saved < len(order):
        print(f"[提醒] 还差 {len(order)-total_saved} 条（多为尚未下载/missing），"
              f"补齐视频后重跑同命令即可续。", flush=True)
    print("[务必] 把上面的 results jsonl 拷回本地/备份，再做 analyze.py 分析。", flush=True)


if __name__ == "__main__":
    main()

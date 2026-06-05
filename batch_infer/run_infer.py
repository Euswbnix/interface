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

# 让 config / first_step / second_step / interface 可导入（仓库根目录）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm

# 静音 mmengine / mmaction / 第三方库的 logging 噪声
for name in ("", "mmengine", "mmaction", "mmcv"):
    logging.getLogger(name).setLevel(logging.ERROR)

from interface import SliceInference  # noqa: E402

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
    args = ap.parse_args()

    order = load_order(args.manifest)
    done = load_done(args.out)
    todo = [h for h in order if h not in done]
    print(f"清单 {len(order)} 条，已完成 {len(done)} 条，本次待处理 {len(todo)} 条", flush=True)

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
            except Exception as e:  # noqa
                rec.update({"class": None, "score": None, "error": f"{type(e).__name__}: {e}"})
                n_err += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if (n_ok + n_err) % 200 == 0:      # 周期强制落盘，断电也不丢
                os.fsync(out.fileno())
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

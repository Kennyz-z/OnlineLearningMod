#!/usr/bin/env python3
# driver_batch_train.py
# ------------------------------------------------------------
# 多进程一次性训练；无任何绘图逻辑
# ------------------------------------------------------------
from __future__ import annotations

import os
from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Tuple

import numpy as np

from core_online import runonline                 # 你的 Part‑1 函数

# ---------- 全局目录 ----------
METRICS_DIR = './metrics_cache'                  # 曲线存这里
WEIGHT_DIR  = './precip_weight_2024_new'         # 权重存这里
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR,  exist_ok=True)

N_JOBS = max(cpu_count() - 1, 1)                 # 并行进程数

# ======================================================================
# 子进程：训练单个 lead
# ======================================================================
def _train_one_lead(args: Tuple) -> None:
    """
    子进程任务：训练 1 个 lead，落盘曲线 (.npz) + 权重 (.npy)
    """
    (var, lt, start_time, end_time,
     expert_models, alg, hint) = args

    # ---- 构造 target_dates ----
    t0 = (datetime.strptime(start_time, '%Y%m%d') +
          timedelta(days=lt)).strftime('%Y%m%d')
    target_dates = f"{t0}-{end_time}"

    # ---- 调用在线学习 ----
    final_w, its, losses, regret = runonline(
        var, lt, target_dates,
        expert_models=expert_models,
        alg=alg,
        hint=hint,
        return_metrics=True,
        visualize=False
    )
    # ---------- 构造日期数组 ----------
    # its = [0, 1, 2, ...] 对应 issuance_dates，自己根据 start_time 推
    start_date = datetime.strptime(start_time, '%Y%m%d')
    dates = np.array([np.datetime64(start_date + timedelta(days=int(i)), 'D')
                      for i in its])

    # ---- 1) 曲线写盘 (.npz) ----
    np.savez_compressed(
        f'{METRICS_DIR}/{var}_{alg}_{hint}_lead{lt:02d}.npz',
        its=its,
        dates=dates,                 # <── 新增
        losses=losses,
        regret=regret
    )

    # ---- 2) 权重写盘 (.npy) ----
    np.save(
        f'{WEIGHT_DIR}/{var}_weight_{alg}_{hint}{lt:02d}.npy',
        final_w
    )

    print(f"[{alg}-{hint}] lead={lt} ✅  曲线 & 权重已写盘")


# ======================================================================
# 并行训练入口
# ======================================================================
def train_all(
    var: str,
    lead_times: range,
    start_time: str,
    end_time: str,
    *,
    expert_models: str,
    algs: list[str],
    hints: list[str],
    n_jobs: int = N_JOBS
) -> None:
    """
    并行遍历 (alg, hint, lead) 组合，逐个子进程训练
    """
    args_iter = [
        (var, lt, start_time, end_time, expert_models, alg, hint)
        for alg, hint in product(algs, hints)
        for lt in lead_times
    ]

    print(f"Start training: {len(args_iter)} 任务, 并行 {n_jobs} 进程 …")
    with Pool(processes=n_jobs) as pool:
        pool.map(_train_one_lead, args_iter)
    print("✅  全部训练 & 落盘完成")


# ======================================================================
# CLI 入口
# ======================================================================
if __name__ == '__main__':
    train_all(
        var='precip',
        lead_times=range(10, 30),              # 改成 range(10, 31) 即 10–30 天
        start_time='20210107',
        end_time='20220101',
        expert_models='./model_list.txt',
        algs=['dorm','adaptdorm'],
        # hints=['mean_g', 'prev_g'],
        hints=['prev_g'],
        n_jobs=N_JOBS
    )

#!/usr/bin/env python3
# compare_algorithms_vs_lead.py
import os
from datetime import datetime, timedelta
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from core_online import runonline     # 复用你已有的函数

mpl.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "Songti SC"

# ------------------------------------------------------------
# 业务参数 —— 根据需要自行修改
# ------------------------------------------------------------
VAR          = "precip"
START_DATE   = "20210107"
END_DATE     = "20211230"
MODEL_FILE   = "./model_list.txt"
# HINT_TYPE    = ["prev_g", "mean_g"]
HINT_TYPE    = ["None"]
ALGORITHMS   = ["dorm", "adaptdorm"]
LEAD_DAYS    = list(range(20, 21))     # 20,21,...,29
VISUALIZE    = True
OUTPUT_DIR   = "./compare_suanfa"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------------------------

def build_date_range(start: str, end: str, lead: int) -> str:
    """把 START_DATE 往后平移 lead 天，生成 runonline 用的 date_range 字符串"""
    t0 = datetime.strptime(start, "%Y%m%d") + timedelta(days=lead)
    return f"{t0.strftime('%Y%m%d')}-{end}"

def single_run(var: str, lead: int, date_str: str,
               alg: str, hint: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # runonline returns: final_weights, iters, losses, regret
    _, iters, losses, regret = runonline(
        var,
        lead,
        date_str,
        expert_models=MODEL_FILE,
        alg=alg,
        hint=hint,
        return_metrics=True,
        visualize=False
    )
    return iters, losses, regret

def compare_over_leads():
    # ---------- 初始化二维 metrics 字典 ----------
    metrics = {
        alg: {
            hint: {'lead': [], 'mean_loss': [], 'mean_regret': []}
            for hint in HINT_TYPE
        }
        for alg in ALGORITHMS
    }

    # ---------- 主循环 ----------
    for lead in LEAD_DAYS:
        date_range = build_date_range(START_DATE, END_DATE, lead)
        for alg in ALGORITHMS:
            for hint in HINT_TYPE:
                print(f"== Lead {lead:2d}d | {alg} + {hint} ==")
                _, loss_arr, reg_arr = single_run(VAR, lead, date_range, alg, hint)
                m = metrics[alg][hint]
                m['lead'].append(lead)
                m['mean_loss'].append(float(loss_arr.mean()))
                m['mean_regret'].append(float(reg_arr.mean()))

    # ---------- 保存 CSV ----------
    rows = []
    for alg in ALGORITHMS:
        for hint in HINT_TYPE:
            m = metrics[alg][hint]
            for ld, ml, mr in zip(m['lead'], m['mean_loss'], m['mean_regret']):
                rows.append({
                    'algorithm': alg,
                    'hint': hint,
                    'lead_day': ld,
                    'mean_loss': ml,
                    'mean_regret': mr
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "lead_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"✔ 已将所有 lead-day 指标保存到 {csv_path}")

    # ---------- 保存原始 metrics 对象 ----------
    npz_path = os.path.join(OUTPUT_DIR, "raw_metrics.npz")
    np.savez(npz_path, metrics=metrics, allow_pickle=True)
    print(f"✔ 已将原始 metrics dict 保存到 {npz_path}")

    # ---------- 可视化 ----------
    if VISUALIZE:
        # (1) Mean Loss vs Lead Day
        plt.figure(figsize=(8, 5))
        for alg in ALGORITHMS:
            for hint in HINT_TYPE:
                m = metrics[alg][hint]
                plt.plot(
                    m['lead'],
                    m['mean_loss'],
                    marker='o',
                    label=f"{alg}+{hint}  MeanLoss"
                )
        plt.xlabel("Lead Day (天)")
        plt.ylabel("所有迭代损失均值")
        plt.title("不同 Lead Day 下 Mean Loss 对比")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # (2) Mean Regret vs Lead Day
        plt.figure(figsize=(8, 5))
        for alg in ALGORITHMS:
            for hint in HINT_TYPE:
                m = metrics[alg][hint]
                plt.plot(
                    m['lead'],
                    m['mean_regret'],
                    marker='s',
                    label=f"{alg}+{hint}  MeanRegret"
                )
        plt.xlabel("Lead Day (天)")
        plt.ylabel("所有迭代 Regret 均值")
        plt.title("不同 Lead Day 下 Mean Regret 对比")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    compare_over_leads()

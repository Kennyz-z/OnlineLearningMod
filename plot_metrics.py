#!/usr/bin/env python3
# plot_metrics.py
# ------------------------------------------------------------
# 读取 metrics_cache/*.npz → 绘制 Loss / Regret 对比
# ------------------------------------------------------------
from __future__ import annotations

import os
from itertools import product
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

METRICS_DIR = './metrics_cache'                 # 与训练脚本保持一致
mpl.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'Songti SC'

# 内存缓存：metrics_store[(lead, alg, hint)] = (its, losses, regret)
metrics_store: Dict[Tuple[int, str, str],
                    Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


# ======================================================================
# 加载函数
# ======================================================================
def load_metrics(
    var: str,
    lead: int,
    algs: List[str],
    hints: List[str]
) -> None:
    """
    尝试把 (lead, alg, hint) 的 .npz 读进 metrics_store
    """
    for alg, hint in product(algs, hints):
        key = (lead, alg, hint)
        if key in metrics_store:
            continue
        fn = f'{METRICS_DIR}/{var}_{alg}_{hint}_lead{lead:02d}.npz'
        if os.path.exists(fn):
            with np.load(fn) as data:
                metrics_store[key] = (
                    data['its'], data['losses'], data['regret']
                )


# ======================================================================
# 绘图函数
# ======================================================================
def plot_lead(
    var: str,
    lead: int,
    algs: List[str],
    hints: List[str],
    *,
    cmap: mpl.colors.Colormap = plt.cm.tab10
) -> None:
    """
    单独绘制一个 lead 的 Loss / Regret 对比图
    """
    load_metrics(var, lead, algs, hints)
    if not any((lead, alg, hint) in metrics_store
               for alg, hint in product(algs, hints)):
        print(f"[warning] lead={lead} 无曲线数据，跳过")
        return

    fig_loss, ax_loss = plt.subplots()
    fig_reg,  ax_reg  = plt.subplots()

    for idx, (alg, hint) in enumerate(product(algs, hints)):
        key = (lead, alg, hint)
        if key not in metrics_store:
            continue
        its, losses, regret = metrics_store[key]
        color = cmap(idx % cmap.N)
        label = f"{alg}-{hint}"

        ax_loss.plot(its, losses, label=label,
                     linewidth=1.4, color=color)
        ax_reg.plot(its, regret, label=label,
                    linewidth=1.4, color=color)

    for ax, ylab, ttl in [
        (ax_loss, '平均损失', f'Lead {lead} d — Loss'),
        (ax_reg,  'Regret',   f'Lead {lead} d — Regret')
    ]:
        ax.set_xlabel('迭代次数')
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.figure.tight_layout()
        plt.show()
        plt.close(ax.figure)

    print(f"[绘图] lead={lead} 完成")


# ======================================================================
# CLI 入口：可一次性绘多 lead
# ======================================================================
if __name__ == '__main__':
    VAR   = 'precip'
    LEADS = range(10, 30)                  # 改成 range(10, 31) 即 10–30
    ALGS  = ["dorm", "adaptdorm"]
    HINTS = ['mean_g']

    for ld in LEADS:
        plot_lead(VAR, ld, ALGS, HINTS)

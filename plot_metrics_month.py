from pickle import FALSE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle

mpl.rcParams['font.family'] = 'Songti SC'     # 中文环境；英文可改 Times

# ------------------------------------------------------------
# 1. 读取 .npz 曲线  →  (dates, losses)
# ------------------------------------------------------------
def load_curve(path):
    data = np.load(path)
    dates  = pd.to_datetime(data['dates'])
    losses = data['losses'].astype(float)     # 逐时间步 RMSE
    return dates, losses


# ------------------------------------------------------------
# 2. 月度累计 RMSE
# ------------------------------------------------------------
def monthly_cum_rmse(dates: pd.DatetimeIndex,
                     losses: np.ndarray) -> pd.Series:
    """
    返回按月重置的累计 RMSE 序列。
    """
    s = pd.Series(losses, index=dates)
    return s.groupby(s.index.to_period('M')).cumsum()


# ------------------------------------------------------------
# 3. 绘图主函数
# ------------------------------------------------------------
def plot_monthly_cum_rmse(npz_dict,
                          *,
                          title='Monthly cumulative RMSE',
                          quarterly_grid=False):
    """
    Parameters
    ----------
    npz_dict : {label: npz_path}   # 多条算法曲线，键=图例名称
    """

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = cycle(['tab:blue', 'tab:orange', 'tab:green',
                    'tab:red', 'tab:purple', 'tab:brown'])

    for label, path in npz_dict.items():
        dates, losses = load_curve(path)
        series = monthly_cum_rmse(dates, losses)

        ax.plot(series.index, series.values,
                label=f"{label} (mean {losses.mean():.3f})",
                linewidth=1.2,
                color=next(colors))

    # —— 辅助网格：季度首月虚线（可关闭） ——
    if quarterly_grid:
        months = pd.to_datetime(sorted(set(series.index.to_timestamp())))
        for ts in months:
            if ts.month in (1, 4, 7, 10):
                ax.axvline(ts, color='k', linestyle='--', linewidth=0.6)

    ax.set_ylabel('Cumulative RMSE (reset each month)')
    ax.set_title(title)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.grid(True, linestyle=':', linewidth=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. 示例调用
# ------------------------------------------------------------
if __name__ == '__main__':
    algo_npz = {
        'DORM+  lead15': './metrics_cache/precip_dorm_prev_g_lead15.npz',
        'AdaHedgeD lead15': './metrics_cache/precip_adaptdorm_prev_g_lead15.npz'
    }

    plot_monthly_cum_rmse(
        npz_dict=algo_npz,
        title='Lead‑15 Algorithms · Monthly Cumulative RMSE'
    )

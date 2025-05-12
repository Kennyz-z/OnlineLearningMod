#!/usr/bin/env python3
# test_adaptdormplus_lead20-30.py
from datetime import datetime as dt
import os
from datetime import timedelta
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from core_online import runonline
from poold import create
from src.s2s_environment import S2SEnvironment
from src.s2s_hints import S2SHinter
from src.s2s_hint_environment import S2SHintEnvironment
from src.utils.eval_util import get_target_dates                # 根据需求自行实现



def build_date_range(start: str, end: str, lead: int) -> str:
    """把 START_DATE 往后平移 lead 天，生成 runonline 用的 date_range 字符串"""
    t0 = dt.strptime(start, "%Y%m%d") + timedelta(days=lead)
    return f"{t0.strftime('%Y%m%d')}-{end}"
VAR           = "precip"
LEADS         = list(range(20, 31))              # 20‑30 天
START_DATE    = "20220101"
END_DATE      = "20231231"
EXPERT_LIST   = "./model_list.txt"               # 你的专家列表
SAVE_CSV      = "results/adaptdormplus_grid.csv"
os.makedirs("results", exist_ok=True)



# 待搜索的超参
grid = [
    {"beta": 0.95, "alpha": 0.5, "adaptive_mode": "adagrad"},
    {"beta": 0.97, "alpha": 0.3, "adaptive_mode": "adagrad"},
    {"beta": 0.97, "alpha": 0.3, "adaptive_mode": "gv"},
]

records = []
for hp in grid:
    for lead in LEADS:
        learner = create(
            "adaptdorm",
            model_list=np.loadtxt(EXPERT_LIST, dtype=str).tolist(),
            groups=None,
            T=None,
            beta=hp["beta"],
            alpha=hp["alpha"],
            adaptive_mode=hp["adaptive_mode"],
        )

        date_rng = build_date_range(START_DATE, END_DATE, lead)
        _, loss_arr, reg_arr = runonline(
            VAR, lead, date_rng, learner, hint="mean_g"
        )

        rmse_mean   = np.mean(loss_arr)
        regret_mean = np.mean(reg_arr)

        records.append({
            "lead": lead,
            **hp,
            "rmse": rmse_mean,
            "regret": regret_mean,
        })
        print(f"lead={lead:2d}, "
              f"β={hp['beta']}, α={hp['alpha']}, mode={hp['adaptive_mode']}, "
              f"RMSE={rmse_mean:.3f}, Regret={regret_mean:.3f}")

# 汇总保存
df = pd.DataFrame(records)
df.to_csv(SAVE_CSV, index=False)

# 若想绘图，可取消注释
# for lead in LEADS:
#     sub = df[df.lead == lead]
#     plt.plot(sub.rmse.values, label=f"lead {lead}")
# plt.xlabel("grid id"); plt.ylabel("mean RMSE"); plt.legend()
# plt.savefig("results/rmse_grid.png", dpi=150)
# plt.close()

"""
core_online.py
--------------
封装在线学习主流程，计算并可选返回损失 / 后悔值曲线。
"""

import logging
import os
from datetime import timedelta
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from poold import create
from src.s2s_environment import S2SEnvironment
from src.s2s_hints import S2SHinter
from src.s2s_hint_environment import S2SHintEnvironment
from src.utils.eval_util import get_target_dates

# 全局 Matplotlib & logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mpl.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'Songti SC'  # 适配 macOS 中文

# ---------------------------------------------------------------------- #
# 工具函数
# ---------------------------------------------------------------------- #
def load_models(model_string: str) -> List[str]:
    """从文件或逗号串加载专家模型列表。"""
    if os.path.isfile(model_string):
        with open(model_string, 'r') as f:
            return [ln.strip() for ln in f if ln.strip()]
    return model_string.split(',')


def _plot_single_curve(x, y, *, title: str, ylabel: str) -> None:
    """统一的单曲线绘图接口。"""
    plt.figure()
    plt.plot(x, y, linewidth=1.6)
    plt.title(title)
    plt.xlabel('迭代次数')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------- #
# 在线学习主流程
# ---------------------------------------------------------------------- #
def run_online(
    gt_id: str,
    lead_time: int,
    target_date_str: str,
    model_list: List[str],
    algorithm: str,
    hint_types: List[str],
    regret_period: int = 1000,
    visualize: bool = False
) -> Tuple[
    np.ndarray,        # final_weights
    List[int],         # iterations
    List[float],       # losses_history
    np.ndarray,        # weights_array
    List[List[Tuple[int, dict]]]  # all_losses_fb
]:
    """执行在线学习并返回详细轨迹。"""

    # 1. 生成 issuance_dates
    targets = get_target_dates(date_str=target_date_str)
    issuance_dates = [t - timedelta(days=lead_time) for t in targets]
    T = len(issuance_dates)

    # 2. 创建 learner & 环境
    learner = create(algorithm, model_list=model_list, groups=None, T=regret_period)
    env = S2SEnvironment(issuance_dates, model_list, gt_id=gt_id, lead_time=str(lead_time))

    # 3. 创建提示器
    hint_dict = {"default": sorted(hint_types)}
    s2s_hinter = S2SHinter(
        hint_types=hint_dict,
        gt_id=gt_id,
        lead_time=str(lead_time),
        learner=learner,
        environment=env,
        regret_hints=(algorithm != "adahedged"),
        hz_hints=False
    )
    _ = S2SHintEnvironment(
        issuance_dates,
        [f"h_{h}" for h in hint_dict["default"]],
        gt_id=gt_id,
        lead_time=str(lead_time),
        learner=learner
    )

    # 4. 准备日志容器
    iterations: List[int] = []
    losses_history: List[float] = []
    weights_history: List[np.ndarray] = []
    all_losses_fb: List[List[Tuple[int, dict]]] = []

    final_weights = None

    # 5. 主循环
    for t in range(T):
        # 周期重置
        if t > 0 and t % regret_period == 0:
            remain = env.get_losses(
                t,
                os_times=learner.get_outstanding(include=False, all_learners=True),
                override=True
            )
            learner.history.record_losses(remain)
            learner.reset_params(T=regret_period)
            s2s_hinter.reset_hint_data()
            logging.info("已完成第 %d 个周期重置", t // regret_period)

        # 缺报检查
        if not env.check_pred(t):
            logging.warning("第 %d 轮缺少模型预测，跳过", t)
            learner.increment_time()
            continue

        # 获取反馈损失
        os_times = learner.get_outstanding()
        losses_fb = env.get_losses(t, os_times=os_times)
        all_losses_fb.append(losses_fb)

        # 更新提示器
        s2s_hinter.update_hint_data(t, losses_fb)
        processed = {idx for idx, _ in losses_fb}
        hint = s2s_hinter.get_hint(t, [ot for ot in os_times if ot not in processed])

        # 更新 learner
        final_weights = learner.update_and_play(losses_fb, hint)

        # 记录
        iterations.append(t)
        if losses_fb:
            loss_vals = [loss['fun'](w=final_weights) for _, loss in losses_fb]
            losses_history.append(float(np.mean(loss_vals)))
        else:
            losses_history.append(0.0)
        weights_history.append(final_weights.copy())

    # 收尾：记录最后一轮损失
    last = env.get_losses(
        T - 1,
        os_times=learner.get_outstanding(include=False, all_learners=True),
        override=True
    )
    learner.history.record_losses(last)

    # 堆叠权重历史
    weights_array = (np.vstack(weights_history)
                     if weights_history else np.empty((0, len(model_list))))

    # 可视化（可选）
    if visualize and iterations:
        _plot_single_curve(iterations, losses_history,
                           title='平均损失随迭代次数变化', ylabel='平均损失')

    return final_weights, iterations, losses_history, weights_array, all_losses_fb


# ---------------------------------------------------------------------- #
# 对外兼容接口：runonline
# ---------------------------------------------------------------------- #
def runonline(
    var: str,
    lead_time: int,
    target_dates: str,
    expert_models: str = './model_list.txt',
    alg: str = 'dormplus',
    hint: str = 'prev_g',
    *,
    return_metrics: bool = False,   # 若为 True 额外返回 (iter, loss, regret)
    visualize: bool = True          # 是否即时画图
) -> Union[np.ndarray,
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    兼容旧接口，同时支持返回曲线数据。
    """
    # 加载模型 & 提示
    models = load_models(expert_models)
    hints = hint.split(',')

    # 执行在线学习
    final_w, iterations, losses_history, weights_array, all_losses_fb = run_online(
        var,
        lead_time,
        target_dates,
        models,
        alg,
        hints,
        regret_period=1000,
        visualize=False  # 由本函数统一控制可视化
    )

    # ---------------- 计算后悔值 ---------------- #
    num_experts = weights_array.shape[1]
    T = len(iterations)
    expert_losses = np.zeros((T, num_experts))
    for t, losses_fb in enumerate(all_losses_fb):
        for j, (_, loss) in enumerate(losses_fb):
            e_j = np.zeros(num_experts)
            e_j[j] = 1.0
            expert_losses[t, j] = loss['fun'](w=e_j)

    cum_alg_loss = np.cumsum(losses_history)
    cum_expert_losses = np.cumsum(expert_losses, axis=0)
    best_cum_expert = np.min(cum_expert_losses, axis=1)
    regret = cum_alg_loss - best_cum_expert
    # ------------------------------------------------ #

    # 绘图
    if visualize:
        _plot_single_curve(iterations, losses_history, title='平均损失曲线', ylabel='平均损失')
        _plot_single_curve(iterations, regret,          title='后悔值曲线', ylabel='Regret')

    if return_metrics:
        return final_w, np.asarray(iterations), np.asarray(losses_history), regret
    return final_w

"""Генерация визуализаций для приложения ЛР-4 (SVM/Pegasos)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -----------------------------------------------------------------------------
# 1. Hinge loss vs другие функции потерь
# -----------------------------------------------------------------------------
def fig_hinge_loss():
    m = np.linspace(-2, 3, 400)
    hinge = np.maximum(0, 1 - m)
    zero_one = (m < 0).astype(float)
    logistic = np.log(1 + np.exp(-m)) / np.log(2)  # /log(2) для нормировки в точке 0
    squared = (1 - m) ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m, hinge, lw=2.5, label="Hinge: $\\max(0,\\, 1-m)$", color="C0")
    ax.plot(m, zero_one, lw=1.5, label="0/1 loss (то, что мы хотим)", color="black", ls=":")
    ax.plot(m, logistic, lw=1.5, label="Logistic: $\\log(1+e^{-m})/\\log 2$", color="C2", ls="--")
    ax.plot(m, np.clip(squared, 0, 4), lw=1.5, label="Squared: $(1-m)^2$", color="C3", ls="-.")
    ax.axvline(1, color="gray", lw=0.8, ls=":")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.annotate("излом\n(нужен субградиент)", xy=(1, 0), xytext=(1.7, 1.2),
                arrowprops=dict(arrowstyle="->", color="C0"), fontsize=9, color="C0")
    ax.annotate("граница\nклассификации", xy=(0, 1), xytext=(-1.8, 2.5),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax.set_xlabel("margin $m = y \\cdot (w \\cdot x + b)$")
    ax.set_ylabel("loss")
    ax.set_title("Hinge loss как выпуклая верхняя оценка 0/1-loss")
    ax.set_ylim(-0.1, 4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "hinge_loss.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. Геометрия SVM: разделяющая гиперплоскость и margin
# -----------------------------------------------------------------------------
def fig_svm_geometry():
    rng = np.random.default_rng(42)
    n = 18
    pos = rng.normal([2.5, 2.5], 0.55, (n, 2))
    neg = rng.normal([-1.5, -1.5], 0.55, (n, 2))

    # Известная граница: x1 + x2 = 0.5
    w = np.array([1.0, 1.0]) / np.sqrt(2)
    b = -0.5 / np.sqrt(2)
    norm_w = np.linalg.norm(w)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pos[:, 0], pos[:, 1], c="C3", marker="o", s=60, edgecolor="black", label="класс $+1$", zorder=3)
    ax.scatter(neg[:, 0], neg[:, 1], c="C0", marker="s", s=60, edgecolor="black", label="класс $-1$", zorder=3)

    xs = np.linspace(-4, 4, 100)
    boundary = (-w[0] * xs - b) / w[1]
    margin_up = (-w[0] * xs - b + 1) / w[1]
    margin_dn = (-w[0] * xs - b - 1) / w[1]

    ax.plot(xs, boundary, "k-", lw=2, label="$w \\cdot x + b = 0$ (граница)")
    ax.plot(xs, margin_up, "k--", lw=1, alpha=0.6, label="$w \\cdot x + b = \\pm 1$ (margin)")
    ax.plot(xs, margin_dn, "k--", lw=1, alpha=0.6)
    ax.fill_between(xs, margin_dn, margin_up, alpha=0.12, color="gray")

    # Стрелка margin width
    p1 = np.array([0.5, 1.5])
    p2 = p1 - w / norm_w**2  # вдоль -w
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="<->", color="C2", lw=2))
    ax.text(0.05, 0.95, "$\\frac{2}{\\|w\\|}$", fontsize=14, color="C2", weight="bold")

    # Вектор w
    centre = np.array([-0.25, -0.25])
    ax.annotate("", xy=centre + w * 0.9, xytext=centre,
                arrowprops=dict(arrowstyle="->", color="C4", lw=2))
    ax.text(centre[0] + 0.7, centre[1] - 0.4, "$w$", fontsize=13, color="C4", weight="bold")

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_title("SVM: максимизация зазора между классами")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "svm_geometry.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. Pegasos: эволюция границы по итерациям
# -----------------------------------------------------------------------------
def fig_pegasos_evolution():
    rng = np.random.default_rng(7)
    n_per = 40
    pos = rng.normal([2, 1.5], 0.7, (n_per, 2))
    neg = rng.normal([-2, -1.5], 0.7, (n_per, 2))
    X = np.vstack([pos, neg])
    y = np.array([1] * n_per + [-1] * n_per)

    snapshots = [1, 10, 100, 1500]
    snaps = {}

    lam = 1.0
    rng_p = np.random.default_rng(0)
    w = np.zeros(2)
    b = 0.0
    for t in range(1, snapshots[-1] + 1):
        eta = 1.0 / (lam * t)
        i = rng_p.integers(len(X))
        m = y[i] * (X[i] @ w + b)
        if m < 1:
            w = (1 - eta * lam) * w + eta * y[i] * X[i]
            b += eta * y[i]
        else:
            w = (1 - eta * lam) * w
        if t in snapshots:
            snaps[t] = (w.copy(), b)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, t in zip(axes, snapshots):
        w_t, b_t = snaps[t]
        ax.scatter(pos[:, 0], pos[:, 1], c="C3", marker="o", s=25, edgecolor="black", lw=0.4)
        ax.scatter(neg[:, 0], neg[:, 1], c="C0", marker="s", s=25, edgecolor="black", lw=0.4)
        xs = np.linspace(-4, 4, 100)
        if abs(w_t[1]) > 1e-6:
            ax.plot(xs, (-w_t[0] * xs - b_t) / w_t[1], "k-", lw=1.8)
            ax.plot(xs, (-w_t[0] * xs - b_t + 1) / w_t[1], "k--", lw=0.8, alpha=0.55)
            ax.plot(xs, (-w_t[0] * xs - b_t - 1) / w_t[1], "k--", lw=0.8, alpha=0.55)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(f"итерация $t = {t}$\n$\\eta = 1/{t}$")
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.suptitle("Pegasos: как граница уточняется по итерациям", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "pegasos_evolution.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. ROC-кривая: что значит сдвигать порог
# -----------------------------------------------------------------------------
def fig_roc_intuition():
    rng = np.random.default_rng(1)
    n = 200
    # Слегка перекрывающиеся распределения
    scores_pos = rng.normal(1.3, 0.8, n)
    scores_neg = rng.normal(-0.3, 0.8, n)
    scores = np.concatenate([scores_pos, scores_neg])
    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    order = np.argsort(-scores)
    s_sorted = scores[order]
    y_sorted = y_true[order]
    tpr = np.concatenate([[0], np.cumsum(y_sorted) / n])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sorted) / n])
    auc = np.trapezoid(tpr, fpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1) Распределения score'ов с тремя порогами
    ax1.hist(scores_neg, bins=25, alpha=0.55, color="C0", label="класс $-$")
    ax1.hist(scores_pos, bins=25, alpha=0.55, color="C3", label="класс $+$")
    for thr, color, name in [(-1.0, "C2", "$\\tau$ низкий\n(много FP)"),
                              (0.5, "C4", "$\\tau$ оптимальный"),
                              (2.0, "C8", "$\\tau$ высокий\n(много FN)")]:
        ax1.axvline(thr, color=color, lw=2)
        ax1.text(thr + 0.05, ax1.get_ylim()[1] * 0.92, name, color=color, fontsize=8.5, weight="bold")
    ax1.set_xlabel("score $w \\cdot x + b$")
    ax1.set_ylabel("число объектов")
    ax1.set_title("Распределение scores. ROC = что будет\nпри пробеге порога $\\tau$")
    ax1.legend(loc="upper right")

    # 2) Собственно ROC
    ax2.plot(fpr, tpr, "C0", lw=2.5, label=f"ROC, AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k:", lw=1, label="случайный (AUC = 0,5)")
    ax2.fill_between(fpr, tpr, alpha=0.15)
    # Отметить точки порогов
    for thr, color, name in [(-1.0, "C2", "$\\tau$ низкий"),
                              (0.5, "C4", "$\\tau$ опт."),
                              (2.0, "C8", "$\\tau$ высокий")]:
        tp = ((scores >= thr) & (y_true == 1)).sum() / n
        fp = ((scores >= thr) & (y_true == 0)).sum() / n
        ax2.scatter(fp, tp, color=color, s=90, zorder=4, edgecolor="black", lw=0.6, label=name)
    ax2.set_xlabel("FPR = FP / (FP+TN)")
    ax2.set_ylabel("TPR = TP / (TP+FN) = Recall")
    ax2.set_title("ROC-кривая")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "roc_intuition.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Learning rate eta_t и его сумма
# -----------------------------------------------------------------------------
def fig_eta_schedule():
    t = np.arange(1, 1000)
    eta = 1.0 / t
    cum = np.cumsum(eta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogy(t, eta, "C0", lw=2)
    ax1.set_xlabel("итерация $t$")
    ax1.set_ylabel("$\\eta_t = 1/(\\lambda t)$, log scale")
    ax1.set_title("Шаг убывает гиперболически")

    ax2.plot(t, cum, "C2", lw=2, label="$\\sum_{s=1}^t \\eta_s$")
    ax2.plot(t, np.log(t) + 0.577, "k:", label="$\\ln t + \\gamma$")
    ax2.set_xlabel("итерация $t$")
    ax2.set_ylabel("кумулятивная сумма шагов")
    ax2.set_title("Сумма расходится как $\\ln t$:\nможем дойти куда угодно, но медленно")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "eta_schedule.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_hinge_loss()
    fig_svm_geometry()
    fig_pegasos_evolution()
    fig_roc_intuition()
    fig_eta_schedule()
    print(f"Saved figures to {OUT}")

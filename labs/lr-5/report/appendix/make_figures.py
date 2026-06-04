"""Генерация визуализаций для приложения ЛР-5 (Gradient Boosting)."""

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------------------------------------------------------
# 1. Сигмоида и log-odds
# -----------------------------------------------------------------------------
def fig_sigmoid_logodds():
    F = np.linspace(-6, 6, 400)
    p = sigmoid(F)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(F, p, "C0", lw=2.5)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axhline(1, color="gray", lw=0.5)
    ax1.axhline(0.5, color="red", lw=0.8, ls=":")
    ax1.axvline(0, color="red", lw=0.8, ls=":")
    ax1.scatter([0], [0.5], color="red", zorder=4, s=50)
    ax1.text(0.3, 0.55, "$\\sigma(0) = 1/2$", color="red")
    ax1.set_xlabel("$F$ (raw score / log-odds)")
    ax1.set_ylabel("$\\sigma(F)$ = $\\Pr(y=1 \\mid x)$")
    ax1.set_title("Сигмоида: $\\mathbb{R} \\to [0, 1]$")
    ax1.set_ylim(-0.05, 1.05)

    # Обратная: log-odds от p
    p2 = np.linspace(0.01, 0.99, 400)
    F2 = np.log(p2 / (1 - p2))
    ax2.plot(p2, F2, "C2", lw=2.5)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0.5, color="red", lw=0.8, ls=":")
    ax2.scatter([0.5], [0], color="red", zorder=4, s=50)
    ax2.text(0.55, 0.4, "$F = \\log\\frac{p}{1-p}$", color="C2")
    ax2.set_xlabel("$p$ = $\\Pr(y=1)$")
    ax2.set_ylabel("$F$ = log-odds")
    ax2.set_title("Log-odds: обратная к сигмоиде")
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT / "sigmoid_logodds.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. Log-loss и его градиент = y - sigma(F)
# -----------------------------------------------------------------------------
def fig_logloss_and_gradient():
    F = np.linspace(-5, 5, 400)
    loss_y1 = -np.log(sigmoid(F))
    loss_y0 = -np.log(1 - sigmoid(F))
    grad_y1 = sigmoid(F) - 1  # dL/dF for y=1
    grad_y0 = sigmoid(F)      # dL/dF for y=0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(F, loss_y1, "C3", lw=2.5, label="$L(y=1, F) = -\\log \\sigma(F)$")
    ax1.plot(F, loss_y0, "C0", lw=2.5, label="$L(y=0, F) = -\\log(1-\\sigma(F))$")
    ax1.set_xlabel("$F$")
    ax1.set_ylabel("log-loss")
    ax1.set_title("Log-loss: уверенные ошибки штрафуются жёстко")
    ax1.set_ylim(0, 5)
    ax1.legend()

    ax2.plot(F, -grad_y1, "C3", lw=2.5, label="$-\\partial L/\\partial F = 1 - \\sigma(F)$  (для $y=1$)")
    ax2.plot(F, -grad_y0, "C0", lw=2.5, label="$-\\partial L/\\partial F = -\\sigma(F)$  (для $y=0$)")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.text(2.5, 0.5, "$r = y - \\sigma(F)$\nостатки", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
    ax2.set_xlabel("$F$")
    ax2.set_ylabel("отрицательный градиент = $r$ (остаток)")
    ax2.set_title("Остатки: куда сдвигать $F$, чтобы loss упал")
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "logloss_and_gradient.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. Decision stump на одномерных данных
# -----------------------------------------------------------------------------
def fig_decision_stump():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 30))
    # Истинные значения: -1 слева, +1 справа от 5, плюс шум
    y_true = np.where(x < 5, -0.7, 0.7) + rng.normal(0, 0.25, 30)

    # Перебираем пороги и считаем MSE
    thresholds = np.linspace(1, 9, 50)
    losses = []
    for t in thresholds:
        mask = x <= t
        cL = y_true[mask].mean() if mask.any() else 0
        cR = y_true[~mask].mean() if (~mask).any() else 0
        pred = np.where(mask, cL, cR)
        losses.append(np.sum((y_true - pred) ** 2))
    best_t = thresholds[np.argmin(losses)]
    best_cL = y_true[x <= best_t].mean()
    best_cR = y_true[x > best_t].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.scatter(x, y_true, c=["C0" if v < 0 else "C3" for v in y_true], s=50, edgecolor="black", lw=0.5, zorder=3)
    ax1.hlines(best_cL, 0, best_t, color="C2", lw=3, label=f"$c_L = {best_cL:.2f}$")
    ax1.hlines(best_cR, best_t, 10, color="C2", lw=3, label=f"$c_R = {best_cR:.2f}$")
    ax1.axvline(best_t, color="black", lw=1.5, ls="--", label=f"порог $t = {best_t:.2f}$")
    ax1.set_xlabel("$x$ (один признак)")
    ax1.set_ylabel("остатки $r$")
    ax1.set_title("Stump приближает остатки\nдвумя константами")
    ax1.legend()

    ax2.plot(thresholds, losses, "C0", lw=2)
    ax2.axvline(best_t, color="C2", lw=2, ls="--", label=f"оптимум: $t = {best_t:.2f}$")
    ax2.scatter([best_t], [min(losses)], color="C2", s=80, zorder=4)
    ax2.set_xlabel("порог $t$")
    ax2.set_ylabel("$\\sum_i (r_i - h(x_i))^2$")
    ax2.set_title("MSE как функция порога\nалгоритм ищет минимум")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUT / "decision_stump.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Эволюция предсказания F в бустинге
# -----------------------------------------------------------------------------
def fig_boosting_evolution():
    rng = np.random.default_rng(0)
    n = 60
    x = np.sort(rng.uniform(0, 10, n))
    # Истина: y=1 для x > 5, y=0 иначе
    y = (x > 5).astype(float)
    # Сделаем чуть шумно: переключим 5% меток
    flip = rng.choice(n, size=3, replace=False)
    y[flip] = 1 - y[flip]

    # Initial
    p = y.mean()
    F0 = np.log(p / (1 - p))
    F = np.full(n, F0)
    lr = 0.3

    snapshots = [0, 1, 3, 10, 30]
    snaps_F = {0: F.copy()}

    for m in range(1, max(snapshots) + 1):
        p_now = sigmoid(F)
        residuals = y - p_now
        # Простой stump
        best_loss = np.inf
        for t in np.linspace(1, 9, 30):
            mask = x <= t
            if mask.all() or not mask.any():
                continue
            cL = residuals[mask].mean()
            cR = residuals[~mask].mean()
            pred = np.where(mask, cL, cR)
            loss = np.sum((residuals - pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_t, best_cL, best_cR = t, cL, cR
        h = np.where(x <= best_t, best_cL, best_cR)
        F = F + lr * h
        if m in snapshots:
            snaps_F[m] = F.copy()

    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.6), sharey=True)
    x_fine = np.linspace(0, 10, 400)
    for ax, m in zip(axes, snapshots):
        F_at = snaps_F[m]
        # Интерполируем для гладкого вида
        order = np.argsort(x)
        # Чтобы линия выглядела как ступенчатая, нарисуем её точками
        p_at = sigmoid(F_at)
        ax.scatter(x[y == 0], np.zeros((y == 0).sum()), color="C0", marker="s", s=20, edgecolor="black", lw=0.4, zorder=3)
        ax.scatter(x[y == 1], np.ones((y == 1).sum()), color="C3", marker="o", s=20, edgecolor="black", lw=0.4, zorder=3)
        ax.plot(x[order], p_at[order], "C2", lw=2, drawstyle="steps-post" if m > 0 else "default", label="$\\sigma(F)$")
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.15, 1.15)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("$x$")
        ax.set_title(f"$m = {m}$ деревьев")
    axes[0].set_ylabel("$\\sigma(F)$ и метки")
    axes[-1].legend(loc="center right", fontsize=8)
    fig.suptitle("Эволюция вероятности $\\sigma(F(x))$ по числу деревьев", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "boosting_evolution.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Residuals как градиент: где они большие
# -----------------------------------------------------------------------------
def fig_residuals_field():
    F = np.linspace(-4, 4, 100)
    # Для y=1
    r_y1 = 1 - sigmoid(F)
    # Для y=0
    r_y0 = 0 - sigmoid(F)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(F, r_y1, "C3", lw=2.5, label="$y_i = 1$:  $r = 1 - \\sigma(F)$")
    ax.plot(F, r_y0, "C0", lw=2.5, label="$y_i = 0$:  $r = -\\sigma(F)$")
    ax.axhline(0, color="gray", lw=0.5)

    # Отметим интуитивные точки
    samples = [(-2, 1, "уверенно ошибаемся\n$r \\approx 1$", "C3"),
               (2, 1, "уверенно правы\n$r \\approx 0$", "C3"),
               (-2, 0, "уверенно правы\n$r \\approx 0$", "C0"),
               (2, 0, "уверенно ошибаемся\n$r \\approx -1$", "C0")]
    for F_pt, y_pt, txt, color in samples:
        r_pt = y_pt - sigmoid(F_pt)
        ax.scatter(F_pt, r_pt, color=color, s=80, zorder=4, edgecolor="black", lw=0.7)
        offset = (0.3, 0.15) if r_pt > 0 else (0.3, -0.3)
        ax.annotate(txt, xy=(F_pt, r_pt), xytext=(F_pt + offset[0], r_pt + offset[1]),
                    fontsize=8.5, color=color)

    ax.set_xlabel("текущее предсказание $F$")
    ax.set_ylabel("residual $r = y - \\sigma(F)$")
    ax.set_title("Чем сильнее ошибаемся, тем больше |residual|\n— следующее дерево сильнее сдвинет $F$ в нужную сторону")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "residuals_field.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. Learning rate: маленький vs большой
# -----------------------------------------------------------------------------
def fig_learning_rate():
    # Имитация кривой log-loss для разных lr
    M = 200
    m = np.arange(M)
    # learning rates
    curves = {
        "$\\nu = 1.0$ (слишком жадно)": 0.5 * np.exp(-m / 10) + 0.18 + 0.04 * np.sin(m / 5) * np.exp(-m / 50),
        "$\\nu = 0.3$": 0.55 * np.exp(-m / 30) + 0.12,
        "$\\nu = 0.1$ (наш выбор)": 0.6 * np.exp(-m / 70) + 0.10,
        "$\\nu = 0.01$ (слишком медленно)": 0.65 * (1 - m / 800) + 0.05,
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, vals in curves.items():
        ax.plot(m, vals, lw=2, label=label)
    ax.set_xlabel("число деревьев $m$")
    ax.set_ylabel("log-loss (схематично)")
    ax.set_title("Эффект learning rate $\\nu$: компромисс\nмежду скоростью и качеством")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_rate.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_sigmoid_logodds()
    fig_logloss_and_gradient()
    fig_decision_stump()
    fig_boosting_evolution()
    fig_residuals_field()
    fig_learning_rate()
    print(f"Saved figures to {OUT}")

"""
Generate animated explanation GIF for SARIMAX Dashboard
Shows: Historical → In-sample fit → First Difference → Forecast + CI
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "explanation.gif"

np.random.seed(42)
T = 60          # history points
FORECAST = 20   # forecast points

# --- Fake time-series ---
t_hist = np.arange(T)
base = 70 + 10 * np.sin(2 * np.pi * t_hist / 30)
noise = np.random.normal(0, 4, T)
y_hist = base + noise

# Fitted (smoother)
from scipy.ndimage import uniform_filter1d
y_fitted = uniform_filter1d(y_hist, size=5) + np.random.normal(0, 1.5, T)

# Forecast
t_fc = np.arange(T, T + FORECAST)
fc_mean = np.array([y_hist[-1] + i * 0.3 + np.random.normal(0, 0.5) for i in range(FORECAST)])
fc_mean = uniform_filter1d(np.concatenate([y_hist[-3:], fc_mean]), size=4)[3:]
ci_width = np.linspace(3, 12, FORECAST)
fc_upper = fc_mean + ci_width
fc_lower = fc_mean - ci_width

# First difference
delta_y = np.diff(y_hist)

# --- Color palette ---
C_HIST    = "#06b6d4"
C_FITTED  = "#f87171"
C_FORECAST = "#a855f7"
C_CI      = "#a855f7"
C_DELTA   = "#f59e0b"
C_BG      = "#0f172a"
C_CARD    = "#1e293b"
C_TEXT    = "#e2e8f0"
C_SUB     = "#94a3b8"

TOTAL_FRAMES = 140

fig = plt.figure(figsize=(12, 6.5), facecolor=C_BG)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, hspace=0.5, wspace=0.35)

# --- Subplot layout ---
ax_main = fig.add_subplot(2, 2, (1, 2))   # top full width
ax_diff  = fig.add_subplot(2, 2, 3)        # bottom left
ax_ci    = fig.add_subplot(2, 2, 4)        # bottom right

for ax in [ax_main, ax_diff, ax_ci]:
    ax.set_facecolor(C_CARD)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors=C_SUB, labelsize=7.5)
    ax.xaxis.label.set_color(C_SUB)
    ax.yaxis.label.set_color(C_SUB)

fig.suptitle("📡 Cách đọc iTMS SARIMAX Dashboard",
             color=C_TEXT, fontsize=14, fontweight="bold", y=0.97)

# Pre-create line/fill objects
line_hist,  = ax_main.plot([], [], color=C_HIST, lw=2, label="Thực tế (Actual)")
line_fit,   = ax_main.plot([], [], color=C_FITTED, lw=1.5, ls="--", label="Fitted (In-sample)")
line_fc,    = ax_main.plot([], [], color=C_FORECAST, lw=2, ls="--", label="Dự báo (Forecast)")
fill_ci     = ax_main.fill_between([], [], [], alpha=0.15, color=C_CI, label="CI Band")
vline_split = ax_main.axvline(x=-999, color="#64748b", lw=1.2, ls=":")

ax_main.set_xlim(-2, T + FORECAST + 2)
ax_main.set_ylim(40, 110)
ax_main.set_xlabel("Thời gian (phút)", fontsize=8)
ax_main.set_ylabel("Xe / phút", fontsize=8)
ax_main.legend(loc="upper left", fontsize=7.5, facecolor="#0f172a",
               labelcolor=C_TEXT, edgecolor="#334155")

# Annotation boxes on main chart
ann_insample  = ax_main.annotate("", xy=(0, 0), xytext=(0, 0),
    bbox=dict(boxstyle="round,pad=0.3", fc="#164e63", ec=C_HIST, lw=1.2),
    color=C_HIST, fontsize=8, fontweight="bold", alpha=0)
ann_outsample = ax_main.annotate("", xy=(0, 0), xytext=(0, 0),
    bbox=dict(boxstyle="round,pad=0.3", fc="#3b0764", ec=C_FORECAST, lw=1.2),
    color=C_FORECAST, fontsize=8, fontweight="bold", alpha=0)

# diff chart
bars_diff = ax_diff.bar(t_hist[1:], np.zeros(T - 1),
    color=[C_HIST if d >= 0 else C_FITTED for d in delta_y],
    width=0.8, alpha=0)
ax_diff.axhline(0, color="#475569", lw=0.8)
ax_diff.set_xlim(-2, T + 2)
ax_diff.set_ylim(-20, 20)
ax_diff.set_xlabel("Thời gian (phút)", fontsize=8)
ax_diff.set_ylabel("Δy_t", fontsize=8)
ax_diff.set_title("Vi phân cấp 1: Δy_t = y_t − y_{t−1}",
                  color=C_TEXT, fontsize=8.5, pad=5)

# CI chart
line_fc2,  = ax_ci.plot([], [], color=C_FORECAST, lw=2, ls="--")
fill_ci2   = ax_ci.fill_between([], [], [], alpha=0.2, color=C_CI)
ax_ci.set_xlim(T - 5, T + FORECAST + 2)
ax_ci.set_ylim(40, 110)
ax_ci.set_xlabel("Thời gian (phút)", fontsize=8)
ax_ci.set_ylabel("Xe / phút", fontsize=8)
ax_ci.set_title("Confidence Interval — độ bất định tăng theo horizon",
                color=C_TEXT, fontsize=8.5, pad=5)

ann_ci = ax_ci.annotate("", xy=(T + 5, 75), xytext=(T + 5, 90),
    arrowprops=dict(arrowstyle="->", color=C_CI, lw=1.5),
    color=C_CI, fontsize=8, ha="center", alpha=0)

step_label = fig.text(0.5, 0.005, "", ha="center", va="bottom",
                      color=C_SUB, fontsize=9, style="italic")

STEPS = [
    (0,  20,  "Bước 1 — Dữ liệu lịch sử: lưu lượng xe theo từng phút (đường cyan)"),
    (20, 45,  "Bước 2 — In-sample fit: mô hình SARIMAX khớp dữ liệu quá khứ (đường đỏ đứt)"),
    (45, 70,  "Bước 3 — Vi phân cấp 1: Δy_t = y_t − y_{t−1} | xanh=tăng, đỏ=giảm"),
    (70, 100, "Bước 4 — Dự báo tương lai: forecast + CI band mở rộng theo thời gian"),
    (100,140, "Bước 5 — Dải CI (màu tím): càng xa hiện tại, độ bất định càng lớn"),
]

def fade(frame, start, end):
    if frame <= start: return 0.0
    if frame >= end:   return 1.0
    return (frame - start) / (end - start)

def init():
    return []

def animate(frame):
    f = frame

    # Current step label
    for s_start, s_end, label in STEPS:
        if s_start <= f < s_end:
            step_label.set_text(label)

    # === Step 1: Draw history ===
    n_hist = int(min(f / 20 * T, T))
    line_hist.set_data(t_hist[:n_hist], y_hist[:n_hist])

    # === Step 2: Draw fitted (frame 20-45) ===
    if f >= 20:
        n_fit = int(min((f - 20) / 25 * T, T))
        line_fit.set_data(t_hist[:n_fit], y_fitted[:n_fit])

        alpha_ann = fade(f, 25, 40)
        ann_insample.set_alpha(alpha_ann)
        if alpha_ann > 0:
            ann_insample.set_text("In-sample\nfit region")
            ann_insample.xy = (30, 100)
            ann_insample.xytext = (30, 100)

    # === Step 3: Show diff bars (frame 45-70) ===
    if f >= 45:
        bar_alpha = fade(f, 45, 65)
        for i, bar in enumerate(bars_diff):
            bar.set_height(delta_y[i])
            bar.set_alpha(bar_alpha * 0.85)

    # === Step 4: Forecast line + CI (frame 70-100) ===
    if f >= 70:
        n_fc = int(min((f - 70) / 30 * FORECAST, FORECAST))
        if n_fc > 0:
            line_fc.set_data(t_fc[:n_fc], fc_mean[:n_fc])
            line_fc2.set_data(t_fc[:n_fc], fc_mean[:n_fc])

            vline_split.set_xdata([T, T])

            alpha_fc = fade(f, 70, 90)
            ann_outsample.set_alpha(alpha_fc)
            if alpha_fc > 0:
                ann_outsample.set_text("Out-of-sample\nforecast")
                ann_outsample.xy = (T + 10, 100)
                ann_outsample.xytext = (T + 10, 100)

    # === Step 5: CI band (frame 100-140) ===
    if f >= 95:
        n_fc2 = int(min((f - 70) / 30 * FORECAST, FORECAST))
        ci_alpha = fade(f, 100, 120)
        if n_fc2 > 0:
            # Redraw fill_between (can't update in place easily — redraw)
            global fill_ci, fill_ci2
            try:
                fill_ci.remove()
                fill_ci2.remove()
            except Exception:
                pass
            fill_ci  = ax_main.fill_between(
                t_fc[:n_fc2], fc_lower[:n_fc2], fc_upper[:n_fc2],
                alpha=ci_alpha * 0.2, color=C_CI)
            fill_ci2 = ax_ci.fill_between(
                t_fc[:n_fc2], fc_lower[:n_fc2], fc_upper[:n_fc2],
                alpha=ci_alpha * 0.3, color=C_CI)
            line_fc2.set_data(t_fc[:n_fc2], fc_mean[:n_fc2])

        ann_ci.set_alpha(fade(f, 105, 125))
        ann_ci.set_text(f"Dải CI rộng dần\ntheo horizon")

    return []

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=TOTAL_FRAMES, interval=80, blit=False)

writer = PillowWriter(fps=16)
anim.save(str(OUT), writer=writer, dpi=100)
print(f"Saved: {OUT}")
plt.close()

"""
fnirs_visualize_v2.py  —  Muse S Athena fNIRS Data Visualizer (v2)
====================================================================
Usage:
    python fnirs_visualize_v2.py <output.csv>

Designed for the v2 recorder schema with per-hemisphere processing.

Produces a 6-panel dashboard:
  1. ΔHbO timeseries — LEFT vs RIGHT hemisphere with task markers shaded.
     Motion/saturation flags drawn as tick marks along the bottom.
  2. ΔHbR timeseries — LEFT vs RIGHT hemisphere
  3. Lateralization (L_HbO − R_HbO) — positive = left-dominant
  4. Block-averaged HRF for LEFT hemisphere (mean ± SEM)
  5. Block-averaged HRF for RIGHT hemisphere (mean ± SEM)
  6. Per-block peak ΔHbO: LEFT vs RIGHT side-by-side, with LI annotation
  7. Z-score / activation detection trace
  8. Ambient channel (drift check)

Requires: numpy pandas matplotlib scipy
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import uniform_filter1d

# ── colour palette ──────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_PANEL   = "#161b22"
C_GRID    = "#21262d"
C_TEXT    = "#e6edf3"
C_MUTED   = "#8b949e"
C_LEFT    = "#3fb950"   # green — LEFT hemisphere
C_RIGHT   = "#58a6ff"   # blue  — RIGHT hemisphere
C_HBR_L   = "#f85149"   # red   — LEFT HbR
C_HBR_R   = "#ff8c42"   # orange — RIGHT HbR
C_LAT     = "#bc8cff"   # purple — lateralization
C_TASK    = "#388bfd"   # task-epoch fill
C_AMB     = "#d29922"   # amber
C_MOTION  = "#f85149"
C_SAT     = "#ff7b72"
C_Z       = "#bc8cff"

SAMPLING_RATE = 64       # Hz — must match recorder

# ── helpers ─────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"Timestamp", "L_HbO", "L_HbR", "R_HbO", "R_HbR", "Marker"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: CSV is missing columns: {missing}\n"
            f"Found: {list(df.columns)}\n\n"
            "This visualizer requires the v2 recorder format. If you have a v1 CSV\n"
            "(with DELTA_HbO/DELTA_HbR/DELTA_HbT columns), use the original visualizer."
        )

    df["Time_s"] = df["Timestamp"] - df["Timestamp"].iloc[0]

    # Backfill Mean columns if absent (older v2 csvs)
    if "Mean_HbO" not in df.columns:
        df["Mean_HbO"] = (df["L_HbO"] + df["R_HbO"]) / 2
    if "Mean_HbR" not in df.columns:
        df["Mean_HbR"] = (df["L_HbR"] + df["R_HbR"]) / 2

    return df


def get_task_epochs(df: pd.DataFrame):
    marker  = df["Marker"].values
    times   = df["Time_s"].values
    epochs  = []
    in_task = False
    t_start = 0.0
    for i, m in enumerate(marker):
        if m == 1 and not in_task:
            in_task = True
            t_start = times[i]
        elif m == 0 and in_task:
            in_task = False
            epochs.append((t_start, times[i]))
    if in_task:
        epochs.append((t_start, times[-1]))
    return epochs


def epoch_hrf(df, epochs, signal_col, pre_s=3.0, post_s=25.0):
    """Cut [-pre_s, +post_s] windows around each task onset for one signal column."""
    sr   = SAMPLING_RATE
    pre  = int(pre_s  * sr)
    post = int(post_s * sr)
    n_win = pre + post
    sig  = df[signal_col].values
    times = df["Time_s"].values

    # Optional motion exclusion
    motion = df["Motion"].values if "Motion" in df.columns else np.zeros(len(df), dtype=int)

    trials = []
    rejected = 0
    for (t_on, _) in epochs:
        idx = np.argmin(np.abs(times - t_on))
        i0  = idx - pre
        i1  = idx + post
        if i0 < 0 or i1 > len(sig):
            continue
        # reject trial if more than 10% of window is motion-flagged
        if motion[i0:i1].mean() > 0.1:
            rejected += 1
            continue
        win = sig[i0:i1].copy()
        win -= win[:pre].mean()  # baseline-correct
        trials.append(win)

    if not trials:
        return None, None, rejected
    t_axis = np.linspace(-pre_s, post_s, n_win)
    return t_axis, np.array(trials), rejected


def smooth(x, w=5):
    return uniform_filter1d(x, size=int(w * SAMPLING_RATE))


# ── style ───────────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update({
        "figure.facecolor":  C_BG,
        "axes.facecolor":    C_PANEL,
        "axes.edgecolor":    C_GRID,
        "axes.labelcolor":   C_TEXT,
        "axes.titlecolor":   C_TEXT,
        "axes.grid":         True,
        "grid.color":        C_GRID,
        "grid.linewidth":    0.6,
        "xtick.color":       C_MUTED,
        "ytick.color":       C_MUTED,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "text.color":        C_TEXT,
        "legend.facecolor":  C_PANEL,
        "legend.edgecolor":  C_GRID,
        "legend.fontsize":   7,
        "lines.linewidth":   1.2,
        "font.family":       "monospace",
    })


def shade_epochs(ax, epochs):
    for (t0, t1) in epochs:
        ax.axvspan(t0, t1, color=C_TASK, alpha=0.13, lw=0)
        ax.axvline(t0, color=C_TASK, lw=0.8, alpha=0.5, ls="--")


def draw_quality_ticks(ax, df):
    """Tick marks at the bottom of the axis for motion and saturation flags."""
    if "Motion" not in df.columns:
        return
    t = df["Time_s"].values
    motion_t = t[df["Motion"].values == 1]
    sat_t    = t[df["Saturated"].values == 1] if "Saturated" in df.columns else np.array([])

    ymin, ymax = ax.get_ylim()
    tick_y = ymin + (ymax - ymin) * 0.02
    if len(motion_t):
        ax.scatter(motion_t, np.full_like(motion_t, tick_y),
                   marker="|", color=C_MOTION, s=20, alpha=0.7,
                   label=f"Motion ({len(motion_t)})")
    if len(sat_t):
        ax.scatter(sat_t, np.full_like(sat_t, tick_y * 1.05),
                   marker="|", color=C_SAT, s=20, alpha=0.7,
                   label=f"Saturated ({len(sat_t)})")


# ── panels ──────────────────────────────────────────────────────────────────

def panel_hbo_lr(ax, df, epochs):
    t = df["Time_s"].values
    L = smooth(df["L_HbO"].values)
    R = smooth(df["R_HbO"].values)

    shade_epochs(ax, epochs)
    ax.plot(t, L, color=C_LEFT,  lw=1.3, label="LEFT  ΔHbO")
    ax.plot(t, R, color=C_RIGHT, lw=1.3, label="RIGHT ΔHbO")
    ax.axhline(0, color=C_MUTED, lw=0.5, ls=":")

    ax.set_ylabel("ΔHbO (μM)")
    ax.set_title("ΔHbO — Left vs Right Hemisphere")
    ax.set_xlim(t[0], t[-1])

    draw_quality_ticks(ax, df)
    ax.legend(loc="upper right")


def panel_hbr_lr(ax, df, epochs):
    t = df["Time_s"].values
    L = smooth(df["L_HbR"].values)
    R = smooth(df["R_HbR"].values)

    shade_epochs(ax, epochs)
    ax.plot(t, L, color=C_HBR_L, lw=1.3, label="LEFT  ΔHbR")
    ax.plot(t, R, color=C_HBR_R, lw=1.3, label="RIGHT ΔHbR")
    ax.axhline(0, color=C_MUTED, lw=0.5, ls=":")

    ax.set_ylabel("ΔHbR (μM)")
    ax.set_title("ΔHbR — Left vs Right Hemisphere")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right")


def panel_lateralization(ax, df, epochs):
    t   = df["Time_s"].values
    lat = smooth(df["L_HbO"].values - df["R_HbO"].values)

    shade_epochs(ax, epochs)
    ax.fill_between(t, 0, lat, where=(lat >= 0), color=C_LEFT,  alpha=0.4,
                    label="Left-dominant")
    ax.fill_between(t, 0, lat, where=(lat <  0), color=C_RIGHT, alpha=0.4,
                    label="Right-dominant")
    ax.plot(t, lat, color=C_LAT, lw=1.0)
    ax.axhline(0, color=C_MUTED, lw=0.6)

    ax.set_ylabel("L − R   ΔHbO (μM)")
    ax.set_title("Lateralization  (L_HbO − R_HbO)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right")


def panel_hrf(ax, t_axis, hbo_trials, hbr_trials, hemi_label, n_rejected=0):
    if t_axis is None:
        ax.text(0.5, 0.5, "No usable task blocks\n(all rejected for motion?)",
                ha="center", va="center", color=C_MUTED, transform=ax.transAxes)
        ax.set_title(f"HRF — {hemi_label}")
        return

    def plot_mean_sem(arr, color, label):
        m   = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(len(arr))
        ax.plot(t_axis, m, color=color, lw=1.6, label=label)
        ax.fill_between(t_axis, m - sem, m + sem, color=color, alpha=0.18)

    ax.axvline(0, color=C_MUTED, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color=C_MUTED, lw=0.5, ls=":")
    ax.axvspan(0, t_axis[-1], color=C_TASK, alpha=0.07, lw=0)

    hbo_color = C_LEFT  if hemi_label == "LEFT" else C_RIGHT
    hbr_color = C_HBR_L if hemi_label == "LEFT" else C_HBR_R

    plot_mean_sem(hbo_trials, hbo_color, f"ΔHbO (n={len(hbo_trials)})")
    plot_mean_sem(hbr_trials, hbr_color, f"ΔHbR (n={len(hbr_trials)})")

    title = f"HRF — {hemi_label} hemisphere  (mean ± SEM)"
    if n_rejected:
        title += f"   [{n_rejected} rejected]"
    ax.set_title(title)
    ax.set_xlabel("Time relative to task onset (s)")
    ax.set_ylabel("Δ Concentration (μM)")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(MultipleLocator(1))


def panel_block_peaks_lr(ax, df, epochs):
    if not epochs:
        ax.text(0.5, 0.5, "No task blocks recorded",
                ha="center", va="center", color=C_MUTED, transform=ax.transAxes)
        ax.set_title("Per-Block Peak ΔHbO")
        return

    L_hbo = df["L_HbO"].values
    R_hbo = df["R_HbO"].values
    time  = df["Time_s"].values

    L_peaks, R_peaks, lat_idx, labels = [], [], [], []
    for i, (t0, t1) in enumerate(epochs):
        mask = (time >= t0) & (time <= t1)
        if mask.sum() == 0:
            continue
        Lp = float(L_hbo[mask].max())
        Rp = float(R_hbo[mask].max())
        L_peaks.append(Lp)
        R_peaks.append(Rp)
        # Standard lateralization index: (L - R) / (|L| + |R|)
        denom = abs(Lp) + abs(Rp) + 1e-9
        lat_idx.append((Lp - Rp) / denom)
        labels.append(f"B{i+1}\n({t1-t0:.0f}s)")

    x = np.arange(len(L_peaks))
    w = 0.38
    ax.bar(x - w/2, L_peaks, w, color=C_LEFT,  alpha=0.85, label="LEFT  peak")
    ax.bar(x + w/2, R_peaks, w, color=C_RIGHT, alpha=0.85, label="RIGHT peak")
    ax.axhline(0, color=C_MUTED, lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ΔHbO peak (μM)")
    ax.set_title("Per-Block Peak ΔHbO  (L vs R)  +  Lateralization Index")
    ax.legend(loc="lower right")

    # LI annotations on top
    ymax = max(max(L_peaks), max(R_peaks)) if L_peaks else 0
    for xi, li in zip(x, lat_idx):
        col = C_LEFT if li > 0 else C_RIGHT
        ax.text(xi, ymax * 1.05, f"LI={li:+.2f}",
                ha="center", va="bottom", fontsize=7, color=col)
    # Make sure the y-axis has headroom for the LI text
    ax.set_ylim(top=ymax * 1.18)

def panel_zscore(ax, df, epochs):
    t = df["Time_s"].values
    if "Z_Score" not in df.columns:
        ax.text(0.5, 0.5, "Z_Score column not in CSV",
                ha="center", va="center", color=C_MUTED, transform=ax.transAxes)
        ax.set_title("Activation Z-score")
        return

    z = df["Z_Score"].values
    shade_epochs(ax, epochs)
    ax.plot(t, z, color=C_Z, lw=1.0, label="Z-score")
    ax.axhline( 2.5, color=C_MOTION, lw=0.6, ls="--", alpha=0.6, label="±2.5σ")
    ax.axhline(-2.5, color=C_MOTION, lw=0.6, ls="--", alpha=0.6)
    ax.axhline(0,    color=C_MUTED,  lw=0.5, ls=":")

    if "Activated" in df.columns:
        active_t = t[df["Activated"].values == 1]
        if len(active_t):
            ax.scatter(active_t, np.full_like(active_t, ax.get_ylim()[1] * 0.9),
                       marker=".", color=C_LEFT, s=4, alpha=0.5,
                       label=f"Activated ({len(active_t)} samp)")

    ax.set_ylabel("z")
    ax.set_title("Real-time Activation Detector  (z-score vs rest baseline)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right")


def panel_ambient(ax, df):
    t = df["Time_s"].values
    amb_col = "True_Stray_Light" if "True_Stray_Light" in df.columns else None
    if amb_col is None:
        ax.text(0.5, 0.5, "No ambient column", ha="center", va="center",
                color=C_MUTED, transform=ax.transAxes)
        ax.set_title("Ambient")
        return
    amb = smooth(df[amb_col].values, w=2)
    ax.plot(t, amb, color=C_AMB, lw=1.0, label="True stray light")
    ax.set_ylabel("ADU")
    ax.set_title("Ambient (thermal/light drift check)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right")


# ── stats summary ────────────────────────────────────────────────────────────

def print_summary(df, epochs):
    dur = df["Time_s"].iloc[-1]
    print("\n" + "─" * 60)
    print("  fNIRS SESSION SUMMARY (v2)")
    print("─" * 60)
    print(f"  Duration        : {dur:.1f} s  ({dur/60:.1f} min)")
    print(f"  Samples         : {len(df)}")
    print(f"  Task blocks     : {len(epochs)}")

    # Data quality
    if "Motion" in df.columns:
        motion_pct = 100 * df["Motion"].mean()
        print(f"  Motion-flagged  : {motion_pct:.1f}% of samples", end="")
        if motion_pct > 10:
            print("  ⚠ HIGH")
        else:
            print()
    if "Saturated" in df.columns:
        sat_pct = 100 * df["Saturated"].mean()
        print(f"  Saturated       : {sat_pct:.1f}% of samples", end="")
        if sat_pct > 1:
            print("  ⚠ check headband fit / lighting")
        else:
            print()

    print()
    for i, (t0, t1) in enumerate(epochs):
        mask = (df["Time_s"] >= t0) & (df["Time_s"] <= t1)
        Lp = df["L_HbO"][mask].max()
        Rp = df["R_HbO"][mask].max()
        Lm = df["L_HbO"][mask].mean()
        Rm = df["R_HbO"][mask].mean()
        denom = abs(Lp) + abs(Rp) + 1e-9
        li = (Lp - Rp) / denom
        side = "LEFT " if li > 0.1 else "RIGHT" if li < -0.1 else "BOTH "
        print(f"  Block {i+1:2d} ({t1-t0:5.1f}s)  "
              f"L: peak={Lp:+.3f} mean={Lm:+.3f}   "
              f"R: peak={Rp:+.3f} mean={Rm:+.3f}   "
              f"LI={li:+.2f} ({side})")

    if "Activated" in df.columns:
        n_act = int(df["Activated"].sum())
        if n_act > 0:
            print(f"\n  Detector firings: {n_act} samples ({n_act/SAMPLING_RATE:.1f}s total)")

    print("─" * 60 + "\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"\nUSAGE: python {sys.argv[0]} <output.csv>\n")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path} …")

    df     = load_csv(path)
    epochs = get_task_epochs(df)

    # Per-hemisphere HRFs
    L_t, L_hbo_tr, L_rej = epoch_hrf(df, epochs, "L_HbO", 3.0, 25.0)
    _,   L_hbr_tr, _     = epoch_hrf(df, epochs, "L_HbR", 3.0, 25.0)
    R_t, R_hbo_tr, R_rej = epoch_hrf(df, epochs, "R_HbO", 3.0, 25.0)
    _,   R_hbr_tr, _     = epoch_hrf(df, epochs, "R_HbR", 3.0, 25.0)

    print_summary(df, epochs)

    apply_style()

    fig = plt.figure(figsize=(17, 13))
    fig.patch.set_facecolor(C_BG)  
    fig.suptitle(
        f"Muse S Athena  fNIRS Dashboard", 
        fontsize=11, color=C_TEXT, y=0.995, fontfamily="monospace"
    )
   
   
    # 4 rows × 2 cols
    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        hspace=0.55,
        wspace=0.25,
        left=0.06, right=0.98,
        top=0.96,  bottom=0.05
    )

    ax_hbo  = fig.add_subplot(gs[0, :])   # ΔHbO L/R full width
    ax_hbr  = fig.add_subplot(gs[1, 0])   # ΔHbR L/R
    ax_lat  = fig.add_subplot(gs[1, 1])   # lateralization
    ax_hrfL = fig.add_subplot(gs[2, 0])   # HRF left
    ax_hrfR = fig.add_subplot(gs[2, 1])   # HRF right
    ax_z    = fig.add_subplot(gs[3, 0])   # z-score
    ax_bars = fig.add_subplot(gs[3, 1])   # block peaks L vs R

    panel_hbo_lr        (ax_hbo,  df, epochs)
    panel_hbr_lr        (ax_hbr,  df, epochs)
    panel_lateralization(ax_lat,  df, epochs)
    panel_hrf           (ax_hrfL, L_t, L_hbo_tr, L_hbr_tr, "LEFT",  L_rej)
    panel_hrf           (ax_hrfR, R_t, R_hbo_tr, R_hbr_tr, "RIGHT", R_rej)
    panel_zscore        (ax_z,    df, epochs)
    panel_block_peaks_lr(ax_bars, df, epochs)

    for ax in (ax_hbo, ax_hbr, ax_lat, ax_z):
        ax.set_xlabel("Time (s)")

    out_path = path.replace(".csv", "_dashboard_v2.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    print(f"Dashboard saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

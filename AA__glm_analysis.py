"""
glm_analysis.py  —  First-level GLM analysis for Muse S fNIRS data
====================================================================

WHAT THIS SCRIPT DOES
---------------------
Performs a first-level General Linear Model (GLM) analysis on a single-session
fNIRS recording produced by the Muse S Athena recorder. This is the standard
analysis used in published fNIRS work (Ye et al., 2009; Huppert et al., 2009).

INTUITION
---------
"If the brain is responding to the task, then the measured ΔHbO time series
should look like a delayed, smoothed copy of when the task was on. How well
does it match, and how big is the response?"

Made quantitative in four steps:

1. PREDICT  — build a predicted ΔHbO time course by convolving the task
              boxcar (1 = task on, 0 = rest) with the canonical HRF (a
              double-gamma function that captures the slow rise and fall of
              the hemodynamic response).

2. NUISANCE — add a constant, linear drift, and quadratic drift regressors
              so the task estimate isn't contaminated by slow optical-coupling
              drift.

3. FIT      — ordinary least squares:
                  y = β₀·HRF(task) + β₁·1 + β₂·t + β₃·t² + ε
              β₀ is the task effect, in μM.

4. AR(1) CORRECT — fNIRS samples are autocorrelated (the next 64 Hz sample
              looks almost identical to the previous one because the
              hemodynamic response is slow). OLS assumes independent
              residuals, which inflates t-statistics. We apply Cochrane-
              Orcutt AR(1) pre-whitening so the residuals are approximately
              independent. β stays the same; t and p become defensible.

USAGE
-----
    python glm_analysis.py <csv_file>

CSV must be the v2 recorder format with columns:
    Timestamp, L_HbO, L_HbR, R_HbO, R_HbR, Marker

OUTPUT
------
    - Console table of β, SE, t, p, R², ρ per signal
    - Significance and direction interpretation
    - Lateralization index
    - PNG plot: measured vs fitted vs task component

REFERENCES
----------
Friston et al. 1998 (canonical HRF / SPM)
Ye et al. 2009 (NIRS-SPM)
Huppert et al. 2009 (Homer / time-series methods for fNIRS)
Cochrane & Orcutt 1949 (AR(1) pre-whitening)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln

FS = 64  # Hz — Muse S Athena fNIRS sample rate


# =============================================================================
# CANONICAL HEMODYNAMIC RESPONSE FUNCTION
# =============================================================================
def canonical_hrf(duration_s=32, fs=64,
                  peak_delay=6.0, undershoot_delay=16.0,
                  peak_disp=1.0, undershoot_disp=1.0,
                  ratio=6.0):
    """
    SPM-style double-gamma canonical HRF (Friston et al., 1998).

    Difference of two gamma PDFs:
      - Peak gamma at ~6s (ΔHbO rise)
      - Undershoot gamma at ~16s (post-stimulus dip), divided by ratio
    Output is normalized so peak = 1.0, making β interpretable in μM.
    """
    t = np.arange(0, duration_s, 1.0 / fs)

    def gamma_pdf(t, shape, scale):
        with np.errstate(divide='ignore', invalid='ignore'):
            logpdf = ((shape - 1) * np.log(t + 1e-9) - t / scale
                      - shape * np.log(scale) - gammaln(shape))
        pdf = np.exp(logpdf)
        pdf[t <= 0] = 0
        return pdf

    peak = gamma_pdf(t, peak_delay / peak_disp, peak_disp)
    undershoot = gamma_pdf(t, undershoot_delay / undershoot_disp, undershoot_disp)
    hrf = peak - undershoot / ratio
    return hrf / np.max(hrf)


# =============================================================================
# DESIGN MATRIX
# =============================================================================
def build_design_matrix(marker, fs):
    """
    Assemble GLM regressors.

    Columns:
        0 = task     : Marker (0/1) convolved with canonical HRF, normalized
                       so β₀ is in μM (peak HbO change per unit task).
        1 = constant : intercept.
        2 = linear   : (t - t̄) / σ(t)  → slow drift.
        3 = quadratic: linear²            → drift curvature.

    Drift regressors absorb slow non-task variation (skin warming, headband
    settling), forcing the task column to explain only task-locked variance.
    """
    n = len(marker)
    task = marker.astype(float)
    hrf = canonical_hrf(duration_s=32, fs=fs)
    task_conv = np.convolve(task, hrf, mode='full')[:n]
    if task_conv.max() > 0:
        task_conv = task_conv / task_conv.max()
    t = np.arange(n) / fs
    t_norm = (t - t.mean()) / t.std()
    return np.column_stack([task_conv, np.ones(n), t_norm, t_norm ** 2])


# =============================================================================
# AR(1) PRE-WHITENING (Cochrane-Orcutt)
# =============================================================================
def estimate_ar1(resid):
    """
    Lag-1 autocorrelation: corr(r[1:], r[:-1]). fNIRS residuals at 64 Hz
    typically have ρ ≈ 0.95-0.99.
    """
    r0 = resid[:-1]
    r1 = resid[1:]
    return np.sum(r0 * r1) / (np.sum(r0 * r0) + 1e-12)


def prewhiten(y, X, rho):
    """
    Cochrane-Orcutt transform: subtract ρ × previous sample from y and X.
    If ε_t = ρ ε_{t-1} + u_t, then (y_t − ρ y_{t-1}) = (X_t − ρ X_{t-1}) β + u_t
    where u is approximately white noise.
    """
    return y[1:] - rho * y[:-1], X[1:] - rho * X[:-1]


def fit_glm_ar1(y, X, max_iter=5, tol=1e-4):
    """
    Iterative GLM with AR(1) pre-whitening.
      1) OLS fit to estimate residuals.
      2) Estimate ρ from those residuals.
      3) Pre-whiten y and X with ρ.
      4) Refit OLS on whitened data.
      5) Re-estimate ρ from new residuals; iterate to convergence.

    Returns β, SE, t, p, R², dof, ρ, fitted values (on original scale).
    """
    n, p = X.shape

    # Initial unwhitened OLS
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    rho = estimate_ar1(resid)

    for _ in range(max_iter):
        y_w, X_w = prewhiten(y, X, rho)
        XtX_inv_w = np.linalg.pinv(X_w.T @ X_w)
        beta = XtX_inv_w @ X_w.T @ y_w
        resid_w = y_w - X_w @ beta
        # Re-estimate rho from the whitened residuals — these should be
        # nearly white if rho is correct, so the residual ρ_w is the
        # *correction* we need to add (small if converged).
        rho_w = estimate_ar1(resid_w)
        rho_new = rho + rho_w
        # Clamp to (-0.999, 0.999) to keep the transformation stable
        rho_new = max(-0.999, min(0.999, rho_new))
        if abs(rho_new - rho) < tol:
            rho = rho_new
            break
        rho = rho_new

    # Final whitened fit
    y_w, X_w = prewhiten(y, X, rho)
    XtX_inv_w = np.linalg.pinv(X_w.T @ X_w)
    beta = XtX_inv_w @ X_w.T @ y_w
    resid_w = y_w - X_w @ beta

    dof = len(y_w) - p
    sigma2 = (resid_w @ resid_w) / dof
    se = np.sqrt(np.diag(XtX_inv_w) * sigma2)
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    # R² on original (unwhitened) scale — for interpretability
    # R² computed on the whitened scale — the scale on which β was fit.
    # Computing R² on the original (unwhitened) scale can yield negative
    # values when AR(1) ρ is near 1, because β was optimized for the
    # whitened problem and may fit the unwhitened data worse than the mean.
    y_w_hat = X_w @ beta
    ss_res = np.sum((y_w - y_w_hat) ** 2)
    ss_tot = np.sum((y_w - y_w.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Also keep the unwhitened fitted values for plotting against measured data
    y_hat = X @ beta

    return {'beta': beta, 'se': se, 't': t_stats, 'p': p_values,
            'r2': r2, 'dof': dof, 'rho': rho, 'fitted': y_hat}


# =============================================================================
# MAIN
# =============================================================================
def main(csv_path):
    df = pd.read_csv(csv_path)
    fs = FS
    t_axis = df["Timestamp"].values - df["Timestamp"].values[0]
    marker = df["Marker"].values.astype(int)

    # Drop the first 30s of pre-protocol settling (filters still stabilizing)
    settle_n = 30 * fs
    df_a = df.iloc[settle_n:].reset_index(drop=True)
    marker_a = marker[settle_n:]
    t_a = t_axis[settle_n:] - t_axis[settle_n]

    if not np.any(marker_a == 1):
        print("ERROR: no task blocks (Marker==1) found after the settling window.")
        sys.exit(1)

    X = build_design_matrix(marker_a, fs)
    signals = {'L_HbO': df_a['L_HbO'].values, 'L_HbR': df_a['L_HbR'].values,
               'R_HbO': df_a['R_HbO'].values, 'R_HbR': df_a['R_HbR'].values}

    print("\n" + "=" * 80)
    print("  GLM ANALYSIS  —  Canonical HRF + drift + AR(1) pre-whitening")
    print("=" * 80)
    print(f"  File           : {csv_path}")
    print(f"  Samples        : {len(marker_a)}  ({len(marker_a)/fs:.0f} s after settling)")
    print(f"  Task blocks    : {int(np.sum(np.diff(marker_a) == 1))}")
    print(f"  Design matrix  : task_HRF + constant + linear + quadratic drift")
    print(f"  Method         : Cochrane-Orcutt AR(1) pre-whitening, then OLS")
    print("=" * 80)
    print(f"\n  {'Signal':<8}{'β (μM)':>11}{'SE':>10}{'t':>9}"
          f"{'p':>11}{'R²':>8}{'ρ':>8}{'sig':>7}")
    print("  " + "-" * 72)

    results = {}
    for name, y in signals.items():
        res = fit_glm_ar1(y, X)
        results[name] = res
        b, t_, p_, se_, rho_ = (res['beta'][0], res['t'][0], res['p'][0],
                                 res['se'][0], res['rho'])
        if p_ < 0.001:  sig = "***"
        elif p_ < 0.01: sig = "**"
        elif p_ < 0.05: sig = "*"
        else:           sig = "n.s."
        expected_up = name.endswith("HbO")
        flag = "" if ((b > 0) == expected_up) or p_ > 0.05 else "  (wrong sign)"
        print(f"  {name:<8}{b:>+11.4f}{se_:>10.4f}{t_:>+9.2f}"
              f"{p_:>11.4g}{res['r2']:>8.3f}{rho_:>8.3f}{sig:>7}{flag}")

    print("  " + "-" * 72)
    print(f"  effective dof = {results['L_HbO']['dof']}   |"
          f"  *p<0.05  **p<0.01  ***p<0.001  |  ρ = AR(1) coefficient")
    print("=" * 80)

    # Interpretation
    print("\n  INTERPRETATION")
    print("  " + "-" * 72)
    def describe(name, res, expect_positive):
        b, t_, p_ = res['beta'][0], res['t'][0], res['p'][0]
        if p_ >= 0.05:
            return f"  {name}: no significant task effect (β={b:+.3f}, p={p_:.3f})"
        sign_ok = (b > 0) == expect_positive
        if sign_ok:
            return (f"  {name}: SIGNIFICANT in expected direction "
                    f"(β={b:+.3f} μM, t={t_:+.2f}, p={p_:.4g})")
        return (f"  {name}: significant but WRONG SIGN "
                f"(β={b:+.3f} μM, t={t_:+.2f}, p={p_:.4g}) — "
                f"suggests systemic/extracerebral contribution")
    print(describe("L_HbO", results['L_HbO'], True))
    print(describe("L_HbR", results['L_HbR'], False))
    print(describe("R_HbO", results['R_HbO'], True))
    print(describe("R_HbR", results['R_HbR'], False))

    if results['L_HbO']['p'][0] < 0.05 or results['R_HbO']['p'][0] < 0.05:
        bL, bR = results['L_HbO']['beta'][0], results['R_HbO']['beta'][0]
        li = (bL - bR) / (abs(bL) + abs(bR) + 1e-9)
        side = ("LEFT-dominant"  if li > 0.1 else
                "RIGHT-dominant" if li < -0.1 else "bilateral")
        print(f"\n  Lateralization Index = {li:+.2f}  →  {side}")
    print("=" * 80)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    plt.style.use('dark_background')
    #plt.style.use('bmh')

    fig.patch.set_facecolor('#1a1a1a')
    m_diff = np.diff(marker_a.astype(int))
    task_starts = (np.where(m_diff == 1)[0] + 1) / fs
    task_ends = (np.where(m_diff == -1)[0] + 1) / fs
    colors = {'L_HbO': '#5fbf5f', 'L_HbR': '#ff6b6b',
              'R_HbO': '#4ea1ff', 'R_HbR': '#ffa500'}

    for ax, (name, y) in zip(axes, signals.items()):
        for s, e in zip(task_starts, task_ends):
            ax.axvspan(s, e, alpha=0.18, color='steelblue', zorder=0)

        res = results[name]
        ax.plot(t_a, y, color=colors[name], lw=0.9, alpha=0.7, label='measured')
        ax.plot(t_a, res['fitted'], color='white', lw=1.6, alpha=0.9,
                label=f'GLM fit (R²={res["r2"]:.3f})')
        task_pred = X[:, 0] * res['beta'][0]
        ax.plot(t_a, task_pred + res['beta'][1], color='magenta', lw=1.2,
                ls='--', alpha=0.7, label=f'task component (β={res["beta"][0]:+.3f})')
        ax.axhline(0, color='white', lw=0.4, ls=':', alpha=0.3)
        ax.set_ylabel(f'{name} (μM)', color='white')
        b, t_, p_ = res['beta'][0], res['t'][0], res['p'][0]
        sig = '***' if p_<0.001 else '**' if p_<0.01 else '*' if p_<0.05 else 'n.s.'
        ax.set_title(
            f'{name}:  β={b:+.4f} μM,  t={t_:+.2f},  p={p_:.4g},  '
            f'ρ(AR1)={res["rho"]:.3f}  [{sig}]', color='white', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_facecolor('#0f0f0f')

        # --- force all axis text to white ---
        ax.tick_params(colors='white', which='both')           # tick numbers (x and y)
        for spine in ax.spines.values():                       # axis border lines
            spine.set_color('white')
        ax.xaxis.label.set_color('white')                      # in case xlabel is set later
        ax.yaxis.label.set_color('white')                      # ylabel re-forced white

    axes[-1].set_xlabel('Time (s)', color='white')
    plt.suptitle('GLM analysis (AR(1) pre-whitened): measured vs predicted',
                 fontsize=13, color='white', y=0.995)
    plt.tight_layout()
    out_path = os.path.basename(csv_path).rsplit('.', 1)[0] + '_glm.png'
    plt.savefig(out_path, dpi=110, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"\n  Plot saved: {out_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <csv_file>")
        sys.exit(1)
    main(sys.argv[1])

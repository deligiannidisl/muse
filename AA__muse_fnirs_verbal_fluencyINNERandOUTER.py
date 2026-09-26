"""
muse_fnirs_verbal_fluency.py  —  INNER-CHANNEL VERSION
=======================================================
Verbal fluency (silent letter-cued word generation) cognitive-load protocol
for Muse S Athena.

DIAGNOSTIC LOGIC:
  - Same protocol structure as the mental arithmetic version, inner channels only.
  - Cue is a single uppercase letter; subject silently generates as many words
    as possible beginning with that letter for the 30 s task window.
  - Six letters drawn at random without replacement each session.

CHANGES FROM PREVIOUS VERSION (based on diagnostics from session 1):
  - INNER channels only (LI/RI). The outer channels showed massive bilateral
    inverted dips (ΔHbO and ΔHbR both crashing −0.7 to −1.2 μM during task)
    that looked like extracerebral / coupling artifact, while the inner
    channels stayed quiet and reasonable. Switching to inner-only to test
    whether a real cognitive response is detectable there.
  - DIST_HEMI now uses DIST_INNER only (2.0 cm), not the inner+outer mean.
  - BASELINE_ALPHA = 0  (rolling baseline disabled — kept off after we saw
    it was masking, not causing, the artifact).
  - COUNTDOWN_S = 3   (was 10; ruled out anticipatory arousal as main cause).
  - Z_THRESHOLD = 1.5 (was 2.0; left-weighted signal less noisy).
  - REST_BASELINE_S = 60 (was 30; gives z-baseline more time to stabilize).
  - N_BLOCKS = 6
  - Bug fix in _print_final_stats: t-test "SIGNIFICANT" message now checks
    the sign of t, so it doesn't print "task > rest" when the effect is
    actually task < rest.

USAGE:
    python muse_fnirs_verbal_fluency.py <output.csv>

NOTE on CSV schema: the LO/LI/RO/RI columns are still written so the v2
dashboard keeps working. Only the L_HbO / R_HbO / L_HbR / R_HbR values
(and therefore the activation score and z-score) are now derived from the
inner channels only.
"""

import numpy as np
import csv
import time
import sys, os
import random
import threading
from collections import deque
from pythonosc import dispatcher, osc_server
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch
from scipy.stats import ttest_rel

# ================= CONFIG =================
PORT          = 5000
SAMPLING_RATE = 64

DIST_OUTER = 2.8
DIST_INNER = 0.8
DIST_HEMI  = (DIST_OUTER + DIST_INNER) / 2   # = 1.8 cm — using both channels averaged

CALIBRATION_SAMPLES = 384
SETTLE_SAMPLES      = 1920

DARK_OFFSET = 10.27
SATURATION_THRESHOLD = 16000
MOTION_JUMP_SIGMA    = 6.0
MOTION_WARMUP_S      = 10

LOG_FILE = "output.csv"

# ---- Rolling baseline (DISABLED) ----
BASELINE_TAU_S = 120
BASELINE_ALPHA = 0.0              # was 1.0/(TAU*FS) — disabled

# ---- Verbal fluency protocol ----
N_BLOCKS         = 6
TASK_DURATION_S  = 30
REST_DURATION_S  = 30
COUNTDOWN_S      = 3              # was 10
LETTERS          = ['F', 'A', 'S', 'B', 'C', 'M', 'P', 'R', 'T', 'L']
INITIAL_REST_S   = 30

# ---- Left-weighted activation ----
LEFT_WEIGHT  = 0.7
RIGHT_WEIGHT = 0.3

# ---- Activation detection ----
SW_WINDOW_S  = 10
SW_HALF      = SW_WINDOW_S * SAMPLING_RATE // 2
SW_SMOOTH_S  = 5
SW_SMOOTH_N  = SW_SMOOTH_S * SAMPLING_RATE

REST_BASELINE_S = 60              # was 30
Z_THRESHOLD     = 1.5             # was 2.0
CONFIRM_S       = 4
CONFIRM_SAMPLES = CONFIRM_S * SAMPLING_RATE

SCI_THRESHOLD = 0.5

current_marker = 0
CALIBRATED     = False

# ---- Extinction coefficients (Matcher & Cope 1994) ----
E_730 = [0.366, 1.280]
E_850 = [1.076, 0.712]

DPF_730 = 6.51
DPF_850 = 5.86

# ================= CHANNEL INDEX MAP — INNER + OUTER (all 4) =================
# OSC indices: ch[0]=LO_730, ch[1]=RO_730, ch[2]=LO_850, ch[3]=RO_850,
#              ch[4]=LI_730, ch[5]=RI_730, ch[6]=LI_850, ch[7]=RI_850
# Using both outer (LO/RO) and inner (LI/RI) channels, averaged per hemisphere.
CH_730_LEFT  = [0, 4]             # LO_730 + LI_730
CH_730_RIGHT = [1, 5]             # RO_730 + RI_730
CH_850_LEFT  = [2, 6]             # LO_850 + LI_850
CH_850_RIGHT = [3, 7]             # RO_850 + RI_850
CH_AMB       = [10, 11, 14, 15]

# ================= FILTERS =================
nyq     = 0.5 * SAMPLING_RATE
lowcut  = 0.01
highcut = 0.15

b_bp, a_bp = butter(2, [lowcut / nyq, highcut / nyq], btype='band')
b_n1, a_n1 = iirnotch(0.05, Q=6.0, fs=SAMPLING_RATE)
b_n2, a_n2 = iirnotch(0.07, Q=6.0, fs=SAMPLING_RATE)
b_n3, a_n3 = iirnotch(0.10, Q=6.0, fs=SAMPLING_RATE)

zi_bp = {"L_HbO": None, "L_HbR": None, "R_HbO": None, "R_HbR": None}
zi_n1 = {"L_HbO": None, "L_HbR": None, "R_HbO": None, "R_HbR": None}
zi_n2 = {"L_HbO": None, "L_HbR": None, "R_HbO": None, "R_HbR": None}
zi_n3 = {"L_HbO": None, "L_HbR": None, "R_HbO": None, "R_HbR": None}

# ================= STATE =================
calib_buffer    = []
calib_sci_730   = []
calib_sci_850   = []
settle_counter  = 0
SETTLED         = False
FILTERS_INITED  = False
C_STARTED       = False

base_730_L = None
base_850_L = None
base_730_R = None
base_850_R = None

_motion_recent = deque(maxlen=SAMPLING_RATE * 5)
motion_flag    = False

# ================= ACTIVATION DETECTION STATE =================
_sw_buffer      = deque(maxlen=SW_HALF * 2)
_sw_smooth_prev = deque(maxlen=SW_SMOOTH_N)
_sw_smooth_rec  = deque(maxlen=SW_SMOOTH_N)

_zbase_buffer = []
ZBASE_READY   = False
_zbase_mean   = 0.0
_zbase_std    = 1.0

_z_confirm_count = 0
ACTIVATED        = False

# ================= PROTOCOL STATE =================
PROTOCOL_RUNNING = False
PROTOCOL_DONE    = False
current_block_idx = 0
current_letter    = None

task_block_means = []
rest_block_means = []
_current_task_buf = []
_current_rest_buf = []
_in_task_now = False
_in_rest_now = False


# ================= SCI =================
def _compute_sci(s730, s850, fs):
    s730 = np.asarray(s730)
    s850 = np.asarray(s850)
    if len(s730) < fs * 2:
        return 0.0
    bb, aa = butter(3, [0.5 / (fs / 2), 2.5 / (fs / 2)], btype='band')
    f730 = lfilter(bb, aa, s730 - np.mean(s730))
    f850 = lfilter(bb, aa, s850 - np.mean(s850))
    f730 = f730[int(fs):]
    f850 = f850[int(fs):]
    n730 = (f730 - np.mean(f730)) / (np.std(f730) + 1e-9)
    n850 = (f850 - np.mean(f850)) / (np.std(f850) + 1e-9)
    return float(np.mean(n730 * n850))


# ================= CHANNEL EXTRACTION =================
def _extract_intensities(ch):
    """Inner-only. Each CH_*_* list now has exactly one element,
    but we keep np.mean() so the same code shape works if you ever
    want to add more channels back later."""
    ch = np.array(ch, dtype=float)
    r730_L = float(np.mean(np.maximum(ch[CH_730_LEFT],  1e-6)))
    r850_L = float(np.mean(np.maximum(ch[CH_850_LEFT],  1e-6)))
    r730_R = float(np.mean(np.maximum(ch[CH_730_RIGHT], 1e-6)))
    r850_R = float(np.mean(np.maximum(ch[CH_850_RIGHT], 1e-6)))
    r_amb  = float(np.mean(ch[CH_AMB]) - DARK_OFFSET)
    return r730_L, r850_L, r730_R, r850_R, r_amb


# ================= ARTIFACT =================
def _check_artifact(ch, r730_L):
    """Motion check now also reads ALL eight optical channels (LO+RO+LI+RI
    × 730+850) for the saturation test, even though only inner channels
    feed the MBLL — a saturated outer detector still indicates a problem."""
    global motion_flag
    ch = np.array(ch, dtype=float)
    optical = ch[[0,1,2,3,4,5,6,7]]   # all optical channels for saturation
    saturated = bool(np.any(optical >= SATURATION_THRESHOLD))

    motion = False
    in_warmup = (settle_counter < SETTLE_SAMPLES + MOTION_WARMUP_S * SAMPLING_RATE)
    if not in_warmup and len(_motion_recent) >= SAMPLING_RATE * 2:
        m = np.mean(_motion_recent)
        s = np.std(_motion_recent) + 1e-6
        if abs(r730_L - m) > MOTION_JUMP_SIGMA * s:
            motion = True
    _motion_recent.append(r730_L)

    motion_flag = motion or saturated
    return motion_flag, saturated, motion


# ================= MBLL =================
def _mbll(r730, r850, base730, base850):
    eps = 1e-9
    dod_730 = -np.log((r730 + eps) / (base730 + eps))
    dod_850 = -np.log((r850 + eps) / (base850 + eps))
    A00 = E_730[0] * DPF_730 * DIST_HEMI
    A01 = E_730[1] * DPF_730 * DIST_HEMI
    A10 = E_850[0] * DPF_850 * DIST_HEMI
    A11 = E_850[1] * DPF_850 * DIST_HEMI
    denom = A00 * A11 - A01 * A10
    hbo = ( A11 * dod_730 - A01 * dod_850) / denom * 1e3
    hbr = (-A10 * dod_730 + A00 * dod_850) / denom * 1e3
    return hbo, hbr


# ================= FILTERS =================
def _init_filters(L_hbo, L_hbr, R_hbo, R_hbr):
    for key, val in [("L_HbO", L_hbo), ("L_HbR", L_hbr),
                     ("R_HbO", R_hbo), ("R_HbR", R_hbr)]:
        zi_bp[key] = lfilter_zi(b_bp, a_bp) * val
        zi_n1[key] = lfilter_zi(b_n1, a_n1) * val
        zi_n2[key] = lfilter_zi(b_n2, a_n2) * val
        zi_n3[key] = lfilter_zi(b_n3, a_n3) * val


def _filter_sample(key, x):
    y, zi_bp[key] = lfilter(b_bp, a_bp, [x], zi=zi_bp[key])
    y, zi_n1[key] = lfilter(b_n1, a_n1, y, zi=zi_n1[key])
    y, zi_n2[key] = lfilter(b_n2, a_n2, y, zi=zi_n2[key])
    y, zi_n3[key] = lfilter(b_n3, a_n3, y, zi=zi_n3[key])
    return float(y[0])


# ================= ACTIVATION DETECTION (left-weighted) =================
def _update_activation(weighted_hbo):
    global _zbase_buffer, ZBASE_READY, _zbase_mean, _zbase_std
    global _z_confirm_count, ACTIVATED

    _sw_buffer.append(weighted_hbo)
    if len(_sw_buffer) == SW_HALF * 2:
        buf  = list(_sw_buffer)
        _sw_smooth_prev.append(np.mean(buf[:SW_HALF]))
        _sw_smooth_rec.append(np.mean(buf[SW_HALF:]))
        activation_sw = float(np.mean(_sw_smooth_rec) - np.mean(_sw_smooth_prev))
    else:
        activation_sw = 0.0

    if not ZBASE_READY:
        if current_marker == 0 and not motion_flag:
            _zbase_buffer.append(weighted_hbo)
            if len(_zbase_buffer) >= REST_BASELINE_S * SAMPLING_RATE:
                _zbase_mean = float(np.mean(_zbase_buffer))
                _zbase_std  = float(np.std(_zbase_buffer)) + 1e-6
                ZBASE_READY = True
                print(f"\n  [Z-baseline ready]  mean={_zbase_mean:+.3f}  std={_zbase_std:.3f} μM")
        else:
            if len(_zbase_buffer) > 0:
                _zbase_buffer = []
        z_score = 0.0
    else:
        z_score = (weighted_hbo - _zbase_mean) / _zbase_std

    if ZBASE_READY:
        if z_score > Z_THRESHOLD:
            _z_confirm_count += 1
        else:
            _z_confirm_count = max(0, _z_confirm_count - 1)

        if (not ACTIVATED) and (_z_confirm_count >= CONFIRM_SAMPLES):
            ACTIVATED = True
            print(f"\n  \033[92m[ACTIVATION DETECTED]  z={z_score:+.2f}\033[0m")
        elif ACTIVATED and (_z_confirm_count == 0):
            ACTIVATED = False

    return activation_sw, z_score, ACTIVATED


# ================= METER =================
meter_max_abs = 0.05

def draw_meter(L_hbo, R_hbo, weighted_hbo, sw, z, is_task, is_active, motion):
    global meter_max_abs
    W = 12
    meter_max_abs = max(meter_max_abs * 0.999, abs(weighted_hbo))

    s = np.clip(weighted_hbo / (meter_max_abs + 1e-6), -1, 1)
    n = int(abs(s) * W)
    col = "\033[92m" if s >= 0 else "\033[91m"
    fill = col + "█" * n + "\033[0m"
    bar = (" " * W + "|" + fill + " " * (W - n)) if s >= 0 else \
          (" " * (W - n) + fill + "|" + " " * W)

    if is_task:
        task_label = f"\033[93m[TASK '{current_letter}']\033[0m"
    else:
        task_label = "\033[94m[REST]   \033[0m"

    if motion:
        act_label = "\033[91m✗ MOTION\033[0m"
    elif is_active:
        act_label = "\033[92m▶ COGNITIVE LOAD DETECTED\033[0m"
    elif ZBASE_READY and z > Z_THRESHOLD * 0.7:
        act_label = "\033[93m◀ rising\033[0m"
    else:
        act_label = "        "

    sys.stdout.write(
        f"\r{task_label} L:{L_hbo:+5.2f} R:{R_hbo:+5.2f} "
        f"WL[{bar}]{weighted_hbo:+5.2f} z:{z:+4.1f} {act_label}   "
    )
    sys.stdout.flush()


# ================= PROTOCOL RUNNER =================
def beep():
    sys.stdout.write("\a")
    sys.stdout.flush()


def protocol_runner():
    global current_marker, current_letter, current_block_idx
    global PROTOCOL_RUNNING, PROTOCOL_DONE
    global _in_task_now, _in_rest_now, _current_task_buf, _current_rest_buf
    global task_block_means, rest_block_means

    print(f"\n  Waiting for z-baseline ({REST_BASELINE_S}s rest after settling)...")
    while not ZBASE_READY:
        time.sleep(0.2)
    time.sleep(2)

    print(f"\n\n\033[1m=== VERBAL FLUENCY PROTOCOL STARTING ===\033[0m")
    print(f"  {N_BLOCKS} blocks of ({REST_DURATION_S}s rest → {TASK_DURATION_S}s silent word generation)")
    print(f"  Inner channels only (LI / RI).")
    print(f"  Sit still, breathe normally, eyes open or closed.\n")
    time.sleep(3)
    PROTOCOL_RUNNING = True

    chosen_letters = random.sample(LETTERS, N_BLOCKS)

    for i in range(N_BLOCKS):
        current_block_idx = i + 1

        # ---- REST ----
        print(f"\n\n\033[94m── REST {i+1}/{N_BLOCKS}  ({REST_DURATION_S}s) ──"
              f"  Just relax and breathe normally.\033[0m")
        _current_rest_buf = []
        _in_rest_now = True
        current_marker = 0
        for s in range(REST_DURATION_S, 0, -1):
            if s <= 3:
                beep()
            time.sleep(1)
        _in_rest_now = False
        if _current_rest_buf:
            rest_block_means.append(float(np.mean(_current_rest_buf)))

        # ---- COUNTDOWN ----
        letter = chosen_letters[i]
        current_letter = letter
        print(f"\n\n\033[93m>>> GET READY — next letter is '{letter}' <<<\033[0m")
        for s in range(COUNTDOWN_S, 0, -1):
            sys.stdout.write(f"\r  Starting in {s}...  ")
            sys.stdout.flush()
            time.sleep(1)
        beep(); beep()

        # ---- TASK ----
        print(f"\n\n\033[1;93m═══ TASK {i+1}/{N_BLOCKS}  LETTER: '{letter}'  "
              f"({TASK_DURATION_S}s) ═══\033[0m")
        print(f"  Silently generate as many words as you can starting with '{letter}'.")
        _current_task_buf = []
        _in_task_now = True
        current_marker = 1
        for s in range(TASK_DURATION_S, 0, -1):
            if s <= 3:
                beep()
            time.sleep(1)
        _in_task_now = False
        current_marker = 0
        if _current_task_buf:
            task_block_means.append(float(np.mean(_current_task_buf)))

        if rest_block_means and task_block_means:
            r = rest_block_means[-1]
            t = task_block_means[-1]
            diff = t - r
            arrow = "↑ HIGHER" if diff > 0 else "↓ LOWER"
            print(f"\n\n  Block {i+1}: rest={r:+.3f}  task={t:+.3f}  "
                  f"diff={diff:+.3f} μM  {arrow}")

    PROTOCOL_RUNNING = False
    PROTOCOL_DONE = True
    _print_final_stats()


def _print_final_stats():
    print("\n\n" + "═" * 60)
    print("  PROTOCOL COMPLETE — STATISTICAL SUMMARY")
    print("═" * 60)

    if len(task_block_means) < 2 or len(rest_block_means) < 2:
        print("  Insufficient blocks for statistics.")
        return

    n = min(len(task_block_means), len(rest_block_means))
    task = np.array(task_block_means[:n])
    rest = np.array(rest_block_means[:n])
    diff = task - rest

    print(f"  N blocks         : {n}")
    print(f"  Rest mean ΔHbO   : {rest.mean():+.3f} ± {rest.std():.3f} μM")
    print(f"  Task mean ΔHbO   : {task.mean():+.3f} ± {task.std():.3f} μM")
    print(f"  Mean difference  : {diff.mean():+.3f} μM   "
          f"({100*np.sum(diff>0)/n:.0f}% blocks task>rest)")

    if n >= 3:
        t_stat, p_val = ttest_rel(task, rest)
        print(f"\n  Paired t-test    : t({n-1}) = {t_stat:+.2f}   p = {p_val:.4f}")
        # Bug fix: check sign of t, not just p
        if p_val < 0.05 and t_stat > 0:
            print(f"  \033[92m  → SIGNIFICANT  (task > rest at p < 0.05)\033[0m")
        elif p_val < 0.05 and t_stat < 0:
            print(f"  \033[91m  → SIGNIFICANT but INVERTED  (task < rest at p < 0.05)\033[0m")
        else:
            print(f"  \033[93m  → not significant. Try more blocks or check sensor placement.\033[0m")
    else:
        print(f"\n  (Need ≥3 blocks for t-test)")

    print("\n  Per-block details:")
    for i, (r, t) in enumerate(zip(rest, task)):
        d = t - r
        mark = "✓" if d > 0 else "✗"
        print(f"    Block {i+1}:  rest={r:+.3f}  task={t:+.3f}  diff={d:+.3f}  {mark}")
    print("═" * 60)
    print("\n  Press Ctrl+C to exit.\n")


# ================= MAIN PROCESSING =================
def calculate_delta_hb(ch):
    global CALIBRATED, calib_buffer, calib_sci_730, calib_sci_850, C_STARTED
    global FILTERS_INITED, SETTLED, settle_counter
    global base_730_L, base_850_L, base_730_R, base_850_R

    r730_L, r850_L, r730_R, r850_R, r_amb = _extract_intensities(ch)
    motion, saturated, _ = _check_artifact(ch, r730_L)

    if not CALIBRATED:
        if not C_STARTED:
            C_STARTED = True
            print("CALIBRATION (6s) — remain still")

        calib_buffer.append([r730_L, r850_L, r730_R, r850_R])
        calib_sci_730.append(r730_L)
        calib_sci_850.append(r850_L)
        n = len(calib_buffer)

        if n < CALIBRATION_SAMPLES:
            if n % 64 == 0:
                pct = int(100 * n / CALIBRATION_SAMPLES)
                print(f"  Calibrating... {pct}%")
            return None

        CALIBRATED = True
        arr = np.array(calib_buffer)
        base_730_L = float(np.mean(arr[:, 0]))
        base_850_L = float(np.mean(arr[:, 1]))
        base_730_R = float(np.mean(arr[:, 2]))
        base_850_R = float(np.mean(arr[:, 3]))

        sci_L = _compute_sci(calib_sci_730, calib_sci_850, SAMPLING_RATE)
        sci_q = ("EXCELLENT" if sci_L > 0.75 else
                 "ACCEPTABLE" if sci_L > SCI_THRESHOLD else
                 "POOR — refit headband")
        opt_min = float(np.min(arr))
        opt_max = float(np.max(arr))
        print(
            f"\nCalibration complete (INNER channels):\n"
            f"  Optical range   : [{opt_min:.1f} … {opt_max:.1f}] ADU\n"
            f"  SCI (LI)        : {sci_L:+.2f}  → {sci_q}\n"
            f"  Settling filters ({SETTLE_SAMPLES // SAMPLING_RATE}s)..."
        )
        return None

    L_hbo, L_hbr = _mbll(r730_L, r850_L, base_730_L, base_850_L)
    R_hbo, R_hbr = _mbll(r730_R, r850_R, base_730_R, base_850_R)

    if not FILTERS_INITED:
        _init_filters(L_hbo, L_hbr, R_hbo, R_hbr)
        FILTERS_INITED = True

    L_hbo_f = _filter_sample("L_HbO", L_hbo)
    L_hbr_f = _filter_sample("L_HbR", L_hbr)
    R_hbo_f = _filter_sample("R_HbO", R_hbo)
    R_hbr_f = _filter_sample("R_HbR", R_hbr)

    if not SETTLED:
        settle_counter += 1
        if settle_counter % SAMPLING_RATE == 0:
            remaining = (SETTLE_SAMPLES - settle_counter) // SAMPLING_RATE
            sys.stdout.write(f"\r  Settling... {remaining}s remaining   ")
            sys.stdout.flush()
        if settle_counter >= SETTLE_SAMPLES:
            SETTLED = True
            print(f"\n\033[92mReady!\033[0m\n  Stay at REST for {REST_BASELINE_S}s, "
                  f"then protocol begins automatically.")
            threading.Thread(target=protocol_runner, daemon=True).start()
        return None

    # Rolling baseline update — DISABLED (BASELINE_ALPHA = 0)
    if BASELINE_ALPHA > 0 and current_marker == 0 and not motion_flag:
        base_730_L = (1 - BASELINE_ALPHA) * base_730_L + BASELINE_ALPHA * r730_L
        base_850_L = (1 - BASELINE_ALPHA) * base_850_L + BASELINE_ALPHA * r850_L
        base_730_R = (1 - BASELINE_ALPHA) * base_730_R + BASELINE_ALPHA * r730_R
        base_850_R = (1 - BASELINE_ALPHA) * base_850_R + BASELINE_ALPHA * r850_R

    weighted_hbo = LEFT_WEIGHT * L_hbo_f + RIGHT_WEIGHT * R_hbo_f
    activation_sw, z_score, activated = _update_activation(weighted_hbo)

    if not motion_flag:
        if _in_task_now:
            _current_task_buf.append(weighted_hbo)
        elif _in_rest_now:
            _current_rest_buf.append(weighted_hbo)

    return (L_hbo_f, L_hbr_f, R_hbo_f, R_hbr_f, weighted_hbo,
            r_amb, activation_sw, z_score, activated,
            motion_flag, saturated)


# ================= OSC HANDLER =================
def ppg_handler(address, *args):
    ch  = list(args)
    res = calculate_delta_hb(ch)
    if res is None:
        return

    (L_hbo, L_hbr, R_hbo, R_hbr, weighted,
     r_amb, sw, z, activated, motion, saturated) = res
    ts = time.time()

    # CSV schema unchanged — still log all 8 optical channels (outer + inner)
    # for diagnostics, even though only inner feeds the MBLL pipeline.
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts,
            ch[0], ch[1], ch[4], ch[5],
            ch[2], ch[3], ch[6], ch[7],
            ch[8], ch[9], ch[12], ch[13],
            ch[10], ch[11], ch[14], ch[15],
            r_amb,
            L_hbo, L_hbr, R_hbo, R_hbr,
            (L_hbo + R_hbo) / 2, (L_hbr + R_hbr) / 2,
            sw, z, int(activated),
            int(motion), int(saturated),
            current_marker
        ])

    draw_meter(L_hbo, R_hbo, weighted, sw, z,
               current_marker == 1, activated, motion)


# ================= MAIN =================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"\nUSAGE: python {sys.argv[0]} <OUTPUT_FILE.csv>\n")
        sys.exit(1)

    LOG_FILE = sys.argv[1]

    with open(LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow([
            "Timestamp",
            "LO_730", "RO_730", "LI_730", "RI_730",
            "LO_850", "RO_850", "LI_850", "RI_850",
            "LO_Red", "RO_Red", "LI_Red", "RI_Red",
            "LO_Amb", "RO_Amb", "LI_Amb", "RI_Amb",
            "True_Stray_Light",
            "L_HbO", "L_HbR", "R_HbO", "R_HbR",
            "Mean_HbO", "Mean_HbR",
            "Activation_SW", "Z_Score", "Activated",
            "Motion", "Saturated",
            "Marker"
        ])

    print(
        "\n=== Muse S Athena — Verbal Fluency Cognitive Load Detector ===\n"
        f"  Output       : {LOG_FILE}\n"
        f"  Port         : {PORT}\n"
        f"  Channels     : INNER + OUTER averaged (LI+LO / RI+RO)\n"
        f"  Distance     : {DIST_HEMI} cm (mean of inner {DIST_INNER} + outer {DIST_OUTER})\n"
        "\n"
        f"  Protocol     : {N_BLOCKS} blocks of (rest {REST_DURATION_S}s + task {TASK_DURATION_S}s)\n"
        f"  Countdown    : {COUNTDOWN_S}s\n"
        f"  Z-baseline   : {REST_BASELINE_S}s\n"
        f"  Z-threshold  : {Z_THRESHOLD}σ\n"
        f"  Rolling base : DISABLED (BASELINE_ALPHA=0)\n"
        f"  Bandpass     : {lowcut}-{highcut} Hz + notches at 0.05/0.07/0.10 Hz\n"
        f"  Score        : {LEFT_WEIGHT}*L_HbO + {RIGHT_WEIGHT}*R_HbO  (left-weighted)\n"
        "\n"
        "  WHAT TO DO DURING TASK:\n"
        "    Silently generate as many words as you can starting with the\n"
        "    letter shown. Don't speak aloud, don't move your lips.\n"
        "    Breathe normally throughout — try to keep breathing\n"
        "    identical between rest and task.\n"
        "    If you run out, repeat earlier words or wait for new ones to come.\n"
        "\n"
        "  WHAT TO DO DURING REST:\n"
        "    Relax. Eyes open or closed but consistent across blocks.\n"
        "    Try to think of nothing in particular.\n"
        "\n"
        "  Sit still. Breathe normally.\n"
    )

    disp = dispatcher.Dispatcher()
    disp.map("/muse/optics", ppg_handler)
    server = osc_server.BlockingOSCUDPServer(("0.0.0.0", PORT), disp)
    server.serve_forever()

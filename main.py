"""gta6-twitch-ccu-forecast-01: hour-by-hour Twitch forecast for GTA VI's launch day.

Model: hour-1-anchored comparable-launch forecaster, v2.

For each of hours 2..24 of the UTC release day the program forecasts five
quantiles of that hour's Twitch concurrent viewers in the GTA VI category.

How it works
------------
The case hands us one anchor: hour 1's print (00:00-01:00 UTC on release
day), plus the hourly curves of 38-42 past big game launches
(`comparables`), each aligned to its own release day's UTC hours.

For every comparable with an hour-1 print we form the log-ratio curve
    r_h = ln(viewers_h / viewers_1),  h = 2..24.
Launch-day shape depends strongly on launch size: a launch whose hour 1
is already huge (global midnight audience) stays roughly flat, while one
whose hour 1 is nearly empty (its main market had not unlocked yet)
explodes later in the day. So each comparable is weighted by how close
its hour-1 level is to the target's, with a Gaussian kernel on log
hour-1 viewers (bandwidth BW). Two centre estimates are then blended
in log space:

  * kernel-weighted median of r_h (regime-aware, follows the size effect);
  * per-hour weighted multi-feature OLS of ln(viewers_h) on
    [1, ln(viewers_1), Thursday?, Friday?], weighted by the same kernel.
    The weekday terms matter because the day curve is a timezone wave:
    a Thursday launch (GTA VI's day) has a dead workday morning and a
    strong post-work evening, while a Friday launch runs into the weekend.

The spread comes from the kernel-weighted quantiles of r_h, recentered
on the blended centre and rescaled toward a blend of the kernel IQR
spread and the OLS residual spread, with a floor and a cap so the
quantiles stay honest (never collapse, never explode). The rescaling is
done separately for the upper and lower halves of the shape, so the
forecast keeps the launches' natural skew (the upper tail is heavier:
a launch can always explode upward far more than it can fall). Both
the median and the spread curves are lightly smoothed across
neighbouring hours.

Everything is standard library only, deterministic, no network, and
runs in well under a second: the same case always writes the same bytes.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only. It reads every <id>.in in
the input directory and writes <id>.out to the output directory:
    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}
exactly 23 rows, hours 2..24 each once, no other key, each q five
finite numbers above 0 and at most 1e9, non-decreasing.

Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a sibling
# module would import; this entry is self-contained, so the line is harmless.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Tunables (chosen on leave-one-out over the comparables; see README) ---
BW = 1.0            # Gaussian kernel bandwidth, in natural-log units of hour-1 viewers
OLS_BLEND = 0.15    # weight on the single-feature log-log OLS centre (fallback only)
OLS2_BLEND = 0.45   # weight on the multi-feature OLS centre (0 = kernel median only)
SPREAD_FLOOR = 0.30  # minimum log-spread (sd) of the forecast distribution
SPREAD_CAP = 0.45    # maximum log-spread (sd)
SHAPE_SCALE_MIN = 0.6  # how far each half of the kernel shape may be shrunk
SHAPE_SCALE_MAX = 1.5  # how far each half of the kernel shape may be stretched


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def comparable_curves(case: dict) -> list[dict]:
    """Release-day curves of the comparables: hour -> viewers, plus hour-1 level.

    A comparable's `day_start` is 00:00 UTC of its release date; its
    `hourly` points cover 14 days before to 3 days after. Only hours
    1..24 of the release day itself are used, and only points with
    viewers > 0 (a log-ratio needs a positive count). Weekday flags come
    from `day_start` (Monday=0 .. Sunday=6).
    """
    curves = []
    for comp in case.get("comparables", []):
        try:
            day_start = _parse_time(comp["day_start"])
        except (KeyError, ValueError):
            continue
        by_hour: dict[int, float] = {}
        for point in comp.get("hourly", []):
            try:
                t = _parse_time(point["from"])
                viewers = float(point["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            if viewers <= 0:
                continue
            hour = int((t - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                by_hour[hour] = viewers
        if 1 not in by_hour:
            continue
        wd = day_start.weekday()
        curves.append({
            "name": comp.get("name", "?"),
            "h1": by_hour[1],
            "log_h1": math.log(by_hour[1]),
            "by_hour": by_hour,
            "thu": 1.0 if wd == 3 else 0.0,
            "fri": 1.0 if wd == 4 else 0.0,
        })
    return curves


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted tau-quantile of values; pairs are (value, weight)."""
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        raise ValueError("weighted_quantile with no weight")
    target = tau * total
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            return value
    return ordered[-1][0]


def ols_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Ordinary least squares y ~ a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    if n < 5:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return (a, b, math.sqrt(var))


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve A x = b by Gaussian elimination with partial pivoting (small n)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for k in range(col, n + 1):
                M[r][k] -= f * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def wols_multi(X: list[list[float]], y: list[float], w: list[float]) -> tuple[list[float], float] | None:
    """Weighted least squares y ~ X (rows include the intercept).

    Returns (beta, residual_sd), or None when too few rows or singular.
    A tiny ridge on the non-intercept diagonals keeps the solve stable.
    """
    p = len(X[0])
    if len(X) < p + 4:
        return None
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for row, yi, wi in zip(X, y, w):
        for i in range(p):
            b[i] += wi * row[i] * yi
            for j in range(p):
                A[i][j] += wi * row[i] * row[j]
    for i in range(1, p):
        A[i][i] += 1e-9
    beta = _solve_linear(A, b)
    if beta is None:
        return None
    resid = [yi - sum(bi * xi for bi, xi in zip(beta, row)) for row, yi in zip(X, y)]
    var = sum(r * r for r in resid) / max(1, len(X) - p)
    return (beta, math.sqrt(var))


def smooth3(values: list[float]) -> list[float]:
    """One pass of [0.25, 0.5, 0.25] smoothing; ends use [0.67, 0.33]."""
    n = len(values)
    if n < 3:
        return list(values)
    out = [0.0] * n
    out[0] = 0.67 * values[0] + 0.33 * values[1]
    out[-1] = 0.33 * values[-2] + 0.67 * values[-1]
    for i in range(1, n - 1):
        out[i] = 0.25 * values[i - 1] + 0.5 * values[i] + 0.25 * values[i + 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    curves = comparable_curves(case)
    if not curves:
        raise ValueError(f"{name} has no usable comparables")

    try:
        target_wd = _parse_time(case["day_start"]).weekday()
    except (KeyError, ValueError):
        target_wd = -1
    target_thu = 1.0 if target_wd == 3 else 0.0
    target_fri = 1.0 if target_wd == 4 else 0.0

    hours = [row["hour"] for row in case["forecast"]]

    # Per-hour: kernel-weighted log-ratio quantiles, multi-feature OLS centre and spread.
    median_log: dict[int, float] = {}
    shape: dict[int, list[float]] = {}
    spread: dict[int, float] = {}
    for h in hours:
        points = []  # (log-ratio, kernel weight, log h1, log viewers_h, curve)
        for curve in curves:
            by_hour = curve["by_hour"]
            if h not in by_hour:
                continue
            log_h1 = curve["log_h1"]
            w = math.exp(-0.5 * ((log_h1 - log_p1) / BW) ** 2)
            points.append((math.log(by_hour[h] / curve["h1"]), w, log_h1, math.log(by_hour[h]), curve))
        if not points:
            # No comparable has this hour: fall back to the previous hour's
            # forecast (hours are processed in order, so it exists unless h is first).
            if h == hours[0]:
                median_log[h] = 0.0
                shape[h] = [Z[i] * 0.45 for i in range(5)]
                spread[h] = 0.45
            else:
                median_log[h] = median_log[h - 1]
                shape[h] = shape[h - 1]
                spread[h] = spread[h - 1]
            continue

        pairs = [(r, w) for (r, w, _, _, _) in points]
        q = [weighted_quantile(pairs, tau) for tau in LEVELS]
        kernel_median = q[2]
        kernel_sd = max(0.05, (q[3] - q[1]) / 1.349)

        # Multi-feature centre: weighted OLS of ln(viewers_h) on
        # [1, ln(viewers_1), Thursday?, Friday?], kernel-weighted.
        centre: float | None = None
        target_sd = kernel_sd
        X = [[1.0, lh, c["thu"], c["fri"]] for (_, _, lh, _, c) in points]
        y = [lv for (_, _, _, lv, _) in points]
        w = [ww for (_, ww, _, _, _) in points]
        mfit = wols_multi(X, y, w)
        if mfit is not None:
            beta, resid_sd = mfit
            tx = [1.0, log_p1, target_thu, target_fri]
            multi_log = sum(bi * xi for bi, xi in zip(beta, tx)) - log_p1
            centre = (1.0 - OLS2_BLEND) * kernel_median + OLS2_BLEND * multi_log
            target_sd = 0.5 * kernel_sd + 0.5 * resid_sd
        else:
            # Fallback: single-feature log-log OLS, as in v1.
            fit = ols_fit([lh for (_, _, lh, _, _) in points], [lv for (_, _, _, lv, _) in points])
            if fit is not None:
                a, b, resid_sd = fit
                ols_log = a + b * log_p1 - log_p1  # OLS centre, as a log-ratio
                centre = (1.0 - OLS_BLEND) * kernel_median + OLS_BLEND * ols_log
                target_sd = 0.5 * kernel_sd + 0.5 * resid_sd
        if centre is None:
            centre = kernel_median
        target_sd = min(SPREAD_CAP, max(SPREAD_FLOOR, target_sd))

        # Recenter the kernel shape on the blended centre, then rescale its
        # upper and lower halves separately toward the target spread
        # (clamped, so the shape keeps its skew).
        centred = [v - kernel_median for v in q]
        up_w = centred[4] / 1.645 if centred[4] > 0 else 0.0
        dn_w = -centred[0] / 1.645 if centred[0] < 0 else 0.0
        s_up = min(SHAPE_SCALE_MAX, max(SHAPE_SCALE_MIN, target_sd / up_w)) if up_w > 1e-9 else 1.0
        s_dn = min(SHAPE_SCALE_MAX, max(SHAPE_SCALE_MIN, target_sd / dn_w)) if dn_w > 1e-9 else 1.0
        median_log[h] = centre
        shape[h] = [centred[i] * (s_dn if i < 2 else (s_up if i > 2 else 1.0)) for i in range(5)]
        spread[h] = target_sd

    # Light smoothing across neighbouring hours (curves, not noise).
    med_curve = smooth3([median_log[h] for h in hours])
    for i, h in enumerate(hours):
        median_log[h] = med_curve[i]
    # Smooth the spread too, via the 5%/95% half-width of each row's shape.
    widths = smooth3([(shape[h][4] - shape[h][0]) / 2.0 for h in hours])
    for i, h in enumerate(hours):
        old = (shape[h][4] - shape[h][0]) / 2.0
        if old > 1e-9:
            factor = widths[i] / old
            shape[h] = [v * factor for v in shape[h]]

    rows = []
    for h in hours:
        quantiles = []
        prev = 0.0
        for i in range(5):
            q = p1 * math.exp(median_log[h] + shape[h][i])
            q = max(1.0, min(1_000_000_000.0, q))
            q = round2(q)
            if q < prev:
                q = prev
            prev = q
            quantiles.append(q)
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

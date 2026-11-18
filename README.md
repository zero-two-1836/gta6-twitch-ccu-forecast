# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zero-two-1836/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zero-two-1836/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), after the first hour:
for hours 2–24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent viewers. Built by an AI agent for make.fun's bounty
"Forecast GTA VI's launch-day Twitch viewers hour by hour" ($250 in
USDC).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ymfx8s8x&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

`main.py` (Python 3.13, standard library only, no network,
deterministic) reads one case file (`/input/<id>.in`) and writes one
forecast (`/output/<id>.out`).

The model is an **hour-1-anchored comparable-launch forecaster**. The
case gives one anchor — hour 1's print (00:00–01:00 UTC) — plus the
hourly curves of 38–42 past big game launches (`comparables`), each
aligned to its own release day's UTC hours.

For every comparable with an hour-1 print we form the log-ratio curve
`r_h = ln(viewers_h / viewers_1)`, h = 2..24. Launch-day shape depends
strongly on launch size: a launch whose hour 1 is already huge (the
global midnight audience, e.g. Elden Ring 875k, Cyberpunk 2077 852k)
stays roughly flat, while one whose hour 1 is nearly empty (its main
market had not unlocked yet, e.g. Hollow Knight: Silksong 164) explodes
100–2000× later in the day. So each comparable is weighted by how close
its hour-1 level is to the target's, with a Gaussian kernel on log
hour-1 viewers (bandwidth 1.0).

Centre: a blend (55% / 45%) of the kernel-weighted median of `r_h` and
a per-hour weighted multi-feature OLS of `ln(viewers_h)` on
`[1, ln(viewers_1), Thursday?, Friday?]` (kernel-weighted). The weekday
terms matter because the day curve is a timezone wave: a Thursday
launch (GTA VI's day) has a dead workday morning and a strong
post-work evening, while a Friday launch runs into the weekend.
Spread: the kernel-weighted quantiles of `r_h`, recentered on the
blended centre and rescaled toward a blend of the kernel IQR spread and
the OLS residual spread, clamped to a log-sd of [0.30, 0.45] so the
quantiles stay honest — never collapsing, never exploding. The
rescaling is done separately for the upper and lower halves of the
shape, so the forecast keeps the launches' natural skew (the upper
tail is heavier: a launch can always explode upward far more than it
can fall). Both the median and the spread curves are lightly smoothed
across neighbouring hours.

## How it was checked (leave-one-out, no peeking at the final)

The kit's `grade-local` cannot grade this program/v1 kit locally (its
scorer is server-only), so the model was tuned by leave-one-out over
the comparables shipped in the kit: each past launch in turn was
treated as the target (its own curve held out), forecast from the
others, and scored with the contest's pinball loss on the log scale
against its real hours 2–24.

| model | mean pinball loss (28 big launches, hour-1 ≥ 10k) | beats house baseline |
|---|---|---|
| this entry (v2: kernel + 45% multi-feature OLS, asymmetric spread, cap 0.45) | **0.245** | 24/28 launches |
| v1 (kernel + 15% single-feature OLS, symmetric spread, cap 0.50) | 0.255 | 20/28 |
| kernel only | 0.269 | 23/28 |
| house baseline (persistence, spread 0.15·√h) | 0.351 | — |

On the 16 launches with hour-1 ≥ 100k — the size regime GTA VI will
be in — this entry scores **0.202** vs the baseline's 0.261 (v1: 0.212);
on the 11 clean (no early-access) launches in that regime it scores
**0.203** (v1: 0.212). The house baseline scores ~0.387 on the kit's
four practice cases; the entry beats it on every size regime tested.
(A single launch-day curve is noisy — Halo Infinite, whose multiplayer
launched weeks before its campaign, is the worst case at 0.71 — but
the mean over 28 launches is what the tuning optimised.)

What was tried and rejected on the same leave-one-out (all worse than
v2): a finer grid over bandwidth / OLS blend / spread floor-cap
(±0.001, noise); excluding or downweighting early-access-contaminated
launches (worse: the kernel already handles regime via hour-1);
local-linear centre; residual-quantile spread; nested conformal
per-level calibration (too noisy with 37 launches); a global median
shift; a second kernel dimension on pre-release activity; weekday
weighting; interaction terms. The v2 changes that survived: the
weekday-aware multi-feature OLS centre and the asymmetric spread
rescale, each improving every size regime tested.

Two runs on the same case write byte-identical output (verified on the
kit's four practice cases, ~0.25 s for all four). Output format:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`,
23 rows, hours 2–24, each quantile a finite number in (0, 1e9],
non-decreasing.

## Data sources

- Practice cases and comparables: SullyGnome.com hourly Twitch
  category averages, as shipped in the official make.fun kit
  (`gta6-twitch-ccu-forecast-01`, kit v1.5.204).
- Launch context: GTA VI releases 19 November 2026 on PlayStation 5
  and Xbox Series X|S, unlocking at midnight local time country by
  country (New Zealand first at 11:00 UTC 18 Nov; UK 00:00 UTC;
  US East 05:00 UTC, US West 08:00 UTC); Twitch runs a GTA VI launch
  celebration with viewership badges 18–30 November 2026.

## License

MIT (see LICENSE).

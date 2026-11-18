"""LG Aimers 9 Phase 2 - inference entry point.

GENERATED FILE - edit src/*.py and rerun `python build_submit.py`.

Every prediction is a function of one evaluation row plus artifacts frozen from
train.csv.  Nothing here groups, sorts, rolls or target-encodes the evaluation
frame, and the calibration shift is a constant baked in at build time.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import json
import pickle
import sys
import time
from math import gcd
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# the merged namespace stands in for the per-module aliases used in src/
C = features = encodings = trackman = calib = sys.modules[__name__]



# ==================== src/config.py ====================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "offer" / "data"
CACHE_DIR = ROOT / "offer" / "cache"          # inside offer/, already gitignored
SUBMIT_DIR = ROOT / "submit"
MODEL_DIR = SUBMIT_DIR / "model"
EXP_DIR = ROOT / "exp"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TRACKMAN_PATH = DATA_DIR / "trackman_history.csv"

TRAIN_ROWS = 1_475_092
TARGET_YEAR = 2025                 # the evaluation season
SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
WALK_FORWARD_YEARS = (2022, 2023, 2024)

TARGET = "control_success"
ROW_ID = "row_id"

# asof_* rate columns that are exact running means over the entity's prior pitches.
# Each maps to the per-pitch binary event it averages (reconstructable by differencing).
PITCHER_RATE_COLUMNS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
BATTER_RATE_COLUMNS = (
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)

# short event names for the reconstructed per-pitch labels
PITCHER_EVENTS = (
    "success", "reverse", "middle", "ball", "strike",
    "fastball", "breaking", "offspeed",
)
BATTER_EVENTS = ("success", "middle")

# The rate columns whose denominator is *not* asof_*_n and therefore hides a count.
PREV_GAME_COLUMNS = {
    1: ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"),
    3: ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev3_game_middle_rate"),
    5: ("asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev5_game_middle_rate"),
}

BASE_STATES = ("___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123")


# ==================== src/features.py ====================

LOGIT_EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------
# 1. base row-wise features
# --------------------------------------------------------------------------

ASOF_COLUMNS = [
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

_BASE_STATE_INDEX = {s: i for i, s in enumerate(C.BASE_STATES)}


def build_base(frame: pd.DataFrame) -> pd.DataFrame:
    f = {}
    g = frame
    f["season"] = g["season"].to_numpy(np.float32)
    f["game_month"] = g["game_month"].to_numpy(np.float32)
    f["game_dayofweek"] = g["game_dayofweek"].to_numpy(np.float32)
    f["inning"] = g["inning"].to_numpy(np.float32)
    f["is_top"] = (g["top_bottom"].to_numpy() == "T").astype(np.float32)
    f["is_regular"] = (g["game_type"].to_numpy() == "R").astype(np.float32)

    balls = g["balls_before"].to_numpy(np.float32)
    strikes = g["strikes_before"].to_numpy(np.float32)
    outs = g["outs_before"].to_numpy(np.float32)
    f["balls_before"] = balls
    f["strikes_before"] = strikes
    f["outs_before"] = outs
    f["count_state"] = balls * 3 + strikes
    f["count_ahead"] = strikes - balls
    f["two_strikes"] = (strikes == 2).astype(np.float32)
    f["three_balls"] = (balls == 3).astype(np.float32)
    f["full_count"] = ((balls == 3) & (strikes == 2)).astype(np.float32)
    f["pitches_in_pa_min"] = balls + strikes          # lower bound on pitch-of-PA

    f["run_total_before"] = g["run_total_before"].to_numpy(np.float32)
    diff = g["score_diff_pitcher_team"].to_numpy(np.float32)
    f["score_diff_pitcher_team"] = diff
    f["abs_score_diff"] = np.abs(diff)
    f["blowout"] = (np.abs(diff) >= 5).astype(np.float32)
    li = g["li"].to_numpy(np.float32)
    f["li"] = li
    f["log_li"] = np.log1p(li)

    home_we = g["home_win_expectancy"].to_numpy(np.float32)
    f["home_win_expectancy"] = home_we
    # the pitching team is the away team when the half-inning is the bottom
    is_top = f["is_top"].astype(bool)
    f["pitcher_team_we"] = np.where(is_top, home_we, 100.0 - home_we).astype(np.float32)

    f["runner_on_1b"] = g["runner_on_1b"].to_numpy(np.float32)
    f["runner_on_2b"] = g["runner_on_2b"].to_numpy(np.float32)
    f["runner_on_3b"] = g["runner_on_3b"].to_numpy(np.float32)
    f["num_runners_on"] = g["num_runners_on"].to_numpy(np.float32)
    f["base_state"] = g["base_state"].map(_BASE_STATE_INDEX).to_numpy(np.float32)
    f["scoring_position"] = ((g["runner_on_2b"] | g["runner_on_3b"])
                             .to_numpy(np.float32))

    p_hand = g["pitcher_hand"].to_numpy(np.float32)
    b_hand = g["batter_hand"].to_numpy(np.float32)
    f["pitcher_hand"] = p_hand
    f["batter_hand"] = b_hand
    f["platoon_same"] = (p_hand == b_hand).astype(np.float32)
    f["hand_pair"] = p_hand * 2 + b_hand
    f["pitcher_id"] = g["pitcher_id"].to_numpy(np.float32)
    f["batter_id"] = g["batter_id"].to_numpy(np.float32)
    f["pitcher_team_id"] = g["pitcher_team_id"].to_numpy(np.float32)
    f["batter_team_id"] = g["batter_team_id"].to_numpy(np.float32)
    f["same_team"] = (g["pitcher_team_id"] == g["batter_team_id"]).to_numpy(np.float32)

    for column in ASOF_COLUMNS:
        f[column] = g[column].to_numpy(np.float32)

    n_p = g["asof_pitcher_n"].to_numpy(np.float64)
    n_b = g["asof_batter_n"].to_numpy(np.float64)
    f["log_pitcher_n"] = np.log1p(n_p).astype(np.float32)
    f["log_batter_n"] = np.log1p(n_b).astype(np.float32)

    p_succ = g["asof_pitcher_success_rate"].to_numpy(np.float64)
    b_succ = g["asof_batter_success_rate"].to_numpy(np.float64)
    f["career_logit_pitcher"] = logit(p_succ).astype(np.float32)
    f["career_logit_batter"] = logit(b_succ).astype(np.float32)
    f["career_logit_sum"] = (f["career_logit_pitcher"] + f["career_logit_batter"])
    f["pitcher_minus_batter"] = (p_succ - b_succ).astype(np.float32)

    for window in (1, 3, 5):
        recent = g[f"asof_pitcher_prev{window}_game_success_rate"].to_numpy(np.float64)
        f[f"form{window}_minus_career"] = (recent - p_succ).astype(np.float32)
        recent_mid = g[f"asof_pitcher_prev{window}_game_middle_rate"].to_numpy(np.float64)
        f[f"formmid{window}_minus_career"] = (
            recent_mid - g["asof_pitcher_middle_rate"].to_numpy(np.float64)).astype(np.float32)
    f["form_trend"] = (
        g["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
        - g["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)).astype(np.float32)

    # repertoire concentration
    mix = np.stack([g["asof_pitcher_fastball_rate"].to_numpy(np.float64),
                    g["asof_pitcher_breaking_rate"].to_numpy(np.float64),
                    g["asof_pitcher_offspeed_rate"].to_numpy(np.float64)])
    with np.errstate(invalid="ignore"):
        f["mix_hhi"] = np.nansum(mix ** 2, axis=0).astype(np.float32)
        f["mix_entropy"] = (-np.nansum(np.where(mix > 0, mix * np.log(np.maximum(mix, 1e-9)), 0.0),
                                       axis=0)).astype(np.float32)
    f["mix_hhi"][np.isnan(mix[0])] = np.nan
    f["mix_entropy"][np.isnan(mix[0])] = np.nan

    out = pd.DataFrame(f, index=frame.index)
    return out.astype(np.float32)


CATEGORICAL_BASE = ["base_state", "count_state", "hand_pair", "pitcher_id", "batter_id",
                    "pitcher_team_id", "batter_team_id"]


# --------------------------------------------------------------------------
# 2. hidden denominators of the prev1/3/5-game rate columns  (axis B1)
# --------------------------------------------------------------------------

def _fraction_table(qmax: int) -> dict[int, int]:
    """value rounded to 6 decimals -> smallest denominator that produces it."""
    table: dict[int, int] = {}
    for q in range(1, qmax + 1):
        for p in range(0, q + 1):
            if gcd(p, q) != 1 and not (p == 0 and q == 1) and not (p == q and q == 1):
                continue
            key = int(round(p / q * 1_000_000))
            if key not in table:
                table[key] = q
    return table


_FRACTIONS: dict[int, int] | None = None
_QMAX = 400


def _lookup(values: np.ndarray) -> np.ndarray:
    global _FRACTIONS
    if _FRACTIONS is None:
        _FRACTIONS = _fraction_table(_QMAX)
    keys = np.where(np.isnan(values), -1, np.round(np.nan_to_num(values) * 1_000_000)).astype(np.int64)
    return pd.Series(keys).map(_FRACTIONS).to_numpy(np.float64)


def prev_game_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover how many pitches each prevN-game window covered.

    The rate columns are p/q rounded to six decimals.  The smallest denominator
    consistent with the printed value divides the true window length; the success
    and middle rates share that denominator, so their least common multiple is a
    sharper estimate (78.5% exact from one column alone, and when wrong it is
    always a proper divisor).
    """
    out = {}
    recovered = {}
    for window, (succ_col, mid_col) in C.PREV_GAME_COLUMNS.items():
        q_s = _lookup(frame[succ_col].to_numpy(np.float64))
        q_m = _lookup(frame[mid_col].to_numpy(np.float64))
        both = ~np.isnan(q_s) & ~np.isnan(q_m)
        q = np.where(np.isnan(q_s), q_m, q_s)
        if both.any():
            a = q_s[both].astype(np.int64)
            b = q_m[both].astype(np.int64)
            lcm = a // np.gcd(a, b) * b
            q[both] = np.where(lcm <= _QMAX, lcm, np.maximum(a, b))
        recovered[window] = q
        out[f"prev{window}_pitches"] = q.astype(np.float32)

    q1, q3, q5 = recovered[1], recovered[3], recovered[5]
    out["prev_pitches_2to3"] = (q3 - q1).astype(np.float32)
    out["prev_pitches_4to5"] = (q5 - q3).astype(np.float32)
    out["prev_pitches_per_game"] = (q5 / 5.0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["prev1_vs_avg"] = (q1 / np.maximum(q5 / 5.0, 1.0)).astype(np.float32)
    out["is_starter_like"] = (q5 / 5.0 >= 45).astype(np.float32)
    out["prev_counts_consistent"] = ((q1 <= q3) & (q3 <= q5)).astype(np.float32)
    # successes implied by the recovered denominators
    for window, (succ_col, _) in C.PREV_GAME_COLUMNS.items():
        out[f"prev{window}_successes"] = (
            frame[succ_col].to_numpy(np.float64) * recovered[window]).astype(np.float32)
    return pd.DataFrame(out, index=frame.index)


# --------------------------------------------------------------------------
# 3. current-season decomposition of the asof running means
# --------------------------------------------------------------------------





def _decompose(frame: pd.DataFrame, joined: pd.DataFrame, key: str, n_col: str,
               rate_cols, event_names, prefix: str, shrink=(25.0, 75.0)) -> pd.DataFrame:
    n_asof = frame[n_col].to_numpy(np.float64)
    n_prev = joined["prev_n"].to_numpy(np.float64)
    n_last = joined["last_n"].to_numpy(np.float64)
    n_std = np.maximum(n_asof - n_prev, 0.0)

    out = {
        f"{prefix}_std_n": n_std,
        f"{prefix}_prev_n": n_prev,
        f"{prefix}_last_n": n_last,
        f"{prefix}_is_debut": (n_prev == 0).astype(np.float64),
        f"{prefix}_season_progress": n_std / (n_last + 50.0),
    }
    for rate_col, event in zip(rate_cols, event_names):
        rate = frame[rate_col].to_numpy(np.float64)
        s_asof = np.where(np.isnan(rate), 0.0, np.round(np.nan_to_num(rate) * n_asof))
        s_prev = joined[f"prev_{prefix}_{event}"].to_numpy(np.float64)
        s_last = joined[f"last_{prefix}_{event}"].to_numpy(np.float64)
        s_std = s_asof - s_prev

        with np.errstate(invalid="ignore", divide="ignore"):
            std_rate = np.where(n_std > 0, s_std / np.maximum(n_std, 1.0), np.nan)
            prev_rate = np.where(n_prev > 0, s_prev / np.maximum(n_prev, 1.0), np.nan)
            last_rate = np.where(n_last > 0, s_last / np.maximum(n_last, 1.0), np.nan)
        out[f"{prefix}_std_{event}"] = std_rate
        out[f"{prefix}_prev_{event}"] = prev_rate
        out[f"{prefix}_last_{event}"] = last_rate
        backoff = np.where(np.isnan(last_rate), np.where(np.isnan(prev_rate), 0.5, prev_rate),
                           last_rate)
        for k in shrink:
            out[f"{prefix}_std_{event}_eb{int(k)}"] = (s_std + k * backoff) / (n_std + k)
        out[f"{prefix}_std_minus_last_{event}"] = std_rate - last_rate
    return pd.DataFrame(out, index=frame.index).astype(np.float32)






def entity_season_inference(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Same decomposition for evaluation rows: a per-row lookup into frozen
    tables, so no evaluation row influences another."""
    blocks = []
    for key, n_col, rate_cols, event_names, prefix, tag in (
        ("pitcher_id", "asof_pitcher_n", C.PITCHER_RATE_COLUMNS, C.PITCHER_EVENTS, "ev_p", "p"),
        ("batter_id", "asof_batter_n", C.BATTER_RATE_COLUMNS, C.BATTER_EVENTS, "ev_b", "b"),
    ):
        table = meta[tag]
        joined = table.reindex(frame[key].to_numpy()).reset_index(drop=True)
        joined = joined.fillna(0.0)
        joined.index = frame.index
        block = _decompose(frame, joined, key, n_col, rate_cols, event_names, prefix)
        block.columns = [c.replace(prefix, tag) for c in block.columns]
        blocks.append(block)
    return pd.concat(blocks, axis=1)


# ==================== src/encodings.py ====================

HALFLIFE = 2.0
SMOOTHING = 200.0

# (name, columns) - all columns must exist on both train and evaluation rows
BASE_KEY_SETS = (
    ("p", ("pitcher_id",)),
    ("b", ("batter_id",)),
    ("p_count", ("pitcher_id", "balls_before", "strikes_before")),
    ("p_bhand", ("pitcher_id", "batter_hand")),
    ("p_runners", ("pitcher_id", "runners_on_flag")),
    ("p_inning", ("pitcher_id", "inning_bucket")),
    ("b_count", ("batter_id", "balls_before", "strikes_before")),
    ("b_phand", ("batter_id", "pitcher_hand")),
    ("pb", ("pitcher_id", "batter_id")),
    ("pteam_count", ("pitcher_team_id", "balls_before", "strikes_before")),
    ("hands_count", ("pitcher_hand", "batter_hand", "balls_before", "strikes_before")),
    ("count_base", ("balls_before", "strikes_before", "base_state_code")),
)

# Deeper interactions, measured and NOT adopted (exp/11).  Adding all eleven moved
# 2022 -23.7, 2024 -3.9 and 2023 +165.7, and the mean looked like a win - but a
# leave-one-out ablation showed removing *any single one* collapsed 2023 back to
# ~549.  That is not how real signal behaves.  The 2023 fold early-stops at 21-25
# trees for twelve of thirteen variants and at 42 for the one that scored 681.7:
# the gain was the early-stopping boundary landing differently, not the features.
# On the two stable folds the base twelve are best.  Kept here so the negative
# result is reproducible rather than re-derived.
EXTRA_KEY_SETS = (
    ("p_count_bhand", ("pitcher_id", "balls_before", "strikes_before", "batter_hand")),
    ("p_base", ("pitcher_id", "base_state_code")),
    ("p_outs", ("pitcher_id", "outs_before")),
    ("p_score", ("pitcher_id", "score_state")),
    ("p_li", ("pitcher_id", "li_bucket")),
    ("p_month", ("pitcher_id", "game_month")),
    ("p_late", ("pitcher_id", "late_inning")),
    ("b_base", ("batter_id", "base_state_code")),
    ("b_count_phand", ("batter_id", "balls_before", "strikes_before", "pitcher_hand")),
    ("bteam_count", ("batter_team_id", "balls_before", "strikes_before")),
    ("hands_base", ("pitcher_hand", "batter_hand", "base_state_code")),
)

KEY_SETS = BASE_KEY_SETS

TARGETS = ("success", "middle", "reverse")


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    extra = pd.DataFrame(index=frame.index)
    extra["runners_on_flag"] = (frame["num_runners_on"].to_numpy() > 0).astype(np.int64)
    extra["inning_bucket"] = np.clip(frame["inning"].to_numpy(), 1, 9).astype(np.int64)
    extra["base_state_code"] = frame["base_state"].map(
        {s: i for i, s in enumerate(C.BASE_STATES)}).to_numpy(np.int64)
    extra["late_inning"] = (frame["inning"].to_numpy() >= 7).astype(np.int64)
    # buckets, not raw values: these keys only work if each cell holds enough rows
    diff = frame["score_diff_pitcher_team"].to_numpy()
    extra["score_state"] = np.digitize(diff, [-3.5, -0.5, 0.5, 3.5]).astype(np.int64)
    extra["li_bucket"] = np.digitize(
        frame["li"].to_numpy(), [0.5, 1.0, 1.5, 2.5]).astype(np.int64)
    return extra










def _encode(agg: pd.DataFrame, code: np.ndarray, smoothing: float) -> dict[str, np.ndarray]:
    out = {}
    if agg.empty:
        return {t: np.full(len(code), np.nan) for t in TARGETS} | {"n": np.zeros(len(code))}
    counts = agg["n"]
    lookup_n = counts.reindex(code).to_numpy(np.float64)
    out["n"] = np.nan_to_num(lookup_n)
    total_n = counts.sum()
    for target in TARGETS:
        # with league-centred targets the shrinkage prior is 0 by construction
        prior = agg[target].sum() / max(total_n, 1e-9)
        sums = np.nan_to_num(agg[target].reindex(code).to_numpy(np.float64))
        out[target] = (sums + smoothing * prior) / (out["n"] + smoothing)
    return out






def transform_encodings(frame: pd.DataFrame, meta: dict, smoothing: float = SMOOTHING) -> pd.DataFrame:
    extra = _augment(frame)
    columns = {}
    for name, key_columns in meta.get("key_sets", KEY_SETS):
        spans = meta["spans"][name]
        code = np.zeros(len(frame), dtype=np.int64)
        valid = np.ones(len(frame), dtype=bool)
        for column, span in zip(key_columns, spans):
            values = (extra[column] if column in extra.columns else frame[column]).to_numpy(np.int64)
            # 2025 contains pitcher and batter ids above every id in train.csv.
            # Clamping them into range would silently attach the highest known
            # entity's history to a debutant, so out-of-range keys are marked
            # invalid and fall back to the smoothing prior instead.
            valid &= (values >= 0) & (values < span)
            code = code * span + np.clip(values, 0, span - 1)
        code = np.where(valid, code, -1)
        encoded = _encode(meta["tables"][name], code, smoothing)
        for target in TARGETS:
            columns[f"te_{name}_{target}"] = encoded[target]
        columns[f"te_{name}_n"] = encoded["n"]
    return pd.DataFrame(columns, index=frame.index).astype(np.float32)


# ==================== src/trackman.py ====================

GROUPS = ("fastball", "breaking", "offspeed")
PHYSICAL = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side", "zone_speed")












PROFILE_PREFIX = "tm_"


def _attach(frame_ids: np.ndarray, mapping: pd.DataFrame, profile: pd.DataFrame,
            index: pd.Index) -> pd.DataFrame:
    link = mapping.set_index("pitcher_id")
    track_id = link["trackman_id"].reindex(frame_ids).to_numpy()
    values = profile.reindex(track_id).reset_index(drop=True)
    values.index = index
    for column in ("map_score", "map_margin", "map_overlap_years", "map_reciprocal"):
        values[column] = link[column].reindex(frame_ids).to_numpy()
    return values.astype(np.float32)






def transform_trackman(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    return _attach(frame["pitcher_id"].to_numpy(), meta["mapping"], meta["profile"],
                   frame.index)


# ==================== src/calib.py ====================

LOGIT_EPS = 1e-7


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def from_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def apply_shift(p: np.ndarray, delta: float) -> np.ndarray:
    return from_logit(to_logit(p) + delta)














# --------------------------------------------------------------------------
# Fixed group-wise logit calibration learned before evaluation inference.
# --------------------------------------------------------------------------





# ==================== src/nninfer.py ====================

# categorical name -> embedding width
NN_CATEGORICALS = {
    "pitcher_id": 32,
    "batter_id": 32,
    "count_state": 4,
    "base_state": 4,
    "hand_pair": 3,
    "pitcher_team_id": 4,
    "batter_team_id": 4,
    "inning": 4,
}


def nn_raw_category(frame: pd.DataFrame, matrix: pd.DataFrame, name: str) -> np.ndarray:
    if name in matrix.columns:
        return matrix[name].to_numpy(np.float64)
    return frame[name].to_numpy(np.float64)


def nn_codes(frame: pd.DataFrame, matrix: pd.DataFrame, bundle: dict) -> np.ndarray:
    names = bundle["names"]
    out = np.zeros((len(matrix), len(names)), dtype=np.int64)
    for i, name in enumerate(names):
        values = nn_raw_category(frame, matrix, name)
        lookup = bundle["maps"][name]
        unseen = len(lookup)
        column = np.full(len(values), unseen, dtype=np.int64)
        finite = np.isfinite(values)
        if finite.any():
            mapped = pd.Series(values[finite]).map(lookup)
            column[finite] = mapped.fillna(unseen).to_numpy(np.int64)
        out[:, i] = column
    return out


def nn_dense(matrix: pd.DataFrame, bundle: dict) -> np.ndarray:
    block = matrix[bundle["dense_columns"]].to_numpy(np.float32)
    missing = np.isnan(block)
    block = (np.nan_to_num(block) - bundle["mean"]) / bundle["std"]
    block = np.clip(block, -8.0, 8.0)
    return np.concatenate([block, missing.astype(np.float32)], axis=1)


def stable_silu(value: np.ndarray) -> np.ndarray:
    """Evaluate SiLU without overflowing exp for large negative activations."""
    value = np.asarray(value)
    sigmoid = np.empty_like(value)
    positive = value >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    sigmoid[~positive] = negative_exp / (1.0 + negative_exp)
    return value * sigmoid


def numpy_forward(bundle: dict, codes: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Run a supported exported neural bundle with NumPy only."""
    state = bundle["state"]
    pieces = [state[f"embeddings.{i}.weight"][codes[:, i]]
              for i in range(len(bundle["names"]))]
    pieces.append(dense)
    x = np.concatenate(pieces, axis=1).astype(np.float32)

    architecture = bundle.get("architecture", "mlp")
    if architecture == "mlp":
        layer = 0
        while f"trunk.{layer}.weight" in state:
            x = x @ state[f"trunk.{layer}.weight"].T + state[f"trunk.{layer}.bias"]
            gamma = state[f"trunk.{layer + 1}.weight"]
            beta = state[f"trunk.{layer + 1}.bias"]
            mean = state[f"trunk.{layer + 1}.running_mean"]
            var = state[f"trunk.{layer + 1}.running_var"]
            x = (x - mean) / np.sqrt(var + 1e-5) * gamma + beta
            x = stable_silu(x)
            layer += 4                       # Linear, BatchNorm, SiLU, Dropout
    elif architecture == "deep_cross":
        original = x
        layer = 0
        while f"cross_v.{layer}.weight" in state:
            cross = x @ state[f"cross_v.{layer}.weight"].T
            cross = stable_silu(cross)
            cross = (cross @ state[f"cross_u.{layer}.weight"].T
                     + state[f"cross_u.{layer}.bias"])
            x = x + original * cross
            gamma = state[f"cross_norm.{layer}.weight"]
            beta = state[f"cross_norm.{layer}.bias"]
            mean = state[f"cross_norm.{layer}.running_mean"]
            var = state[f"cross_norm.{layer}.running_var"]
            x = (x - mean) / np.sqrt(var + 1e-5) * gamma + beta
            x = stable_silu(x)
            layer += 1
        expected_layers = bundle.get("cross_layers")
        if expected_layers is not None and layer != int(expected_layers):
            raise RuntimeError(
                f"deep_cross bundle declares {expected_layers} layers but state has {layer}"
            )
    else:
        raise RuntimeError(f"unsupported neural bundle architecture: {architecture!r}")
    logits = x @ state["head.weight"].T + state["head.bias"]
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 3]


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

# Declared here, after the bundled src/config.py body, because that module
# defines paths of the same name relative to the development checkout.
DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
OUT_PATH = Path("./output/submission.csv")
SUPPORTED_MEMBER_KINDS = {"binary", "multiclass", "catboost", "nn"}


def validate_meta(value: object) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError("model/meta.pkl must contain a dictionary")
    required = {"columns", "members", "shifts", "entity_history", "encodings", "trackman"}
    missing = sorted(required - set(value))
    if missing:
        raise RuntimeError(f"model metadata is missing required keys: {missing}")

    columns = value["columns"]
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(column, str) and column for column in columns)):
        raise RuntimeError("model metadata columns must be a non-empty list of names")
    duplicate_columns = sorted({column for column in columns if columns.count(column) > 1})
    if duplicate_columns:
        raise RuntimeError(f"model metadata has duplicate columns: {duplicate_columns}")

    members = value["members"]
    if not isinstance(members, list) or not members:
        raise RuntimeError("model metadata must contain at least one member")
    files = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise RuntimeError(f"member {index} must be a dictionary")
        member_missing = sorted({"file", "kind", "weight"} - set(member))
        if member_missing:
            raise RuntimeError(f"member {index} is missing keys: {member_missing}")
        kind = member["kind"]
        if kind not in SUPPORTED_MEMBER_KINDS:
            raise RuntimeError(f"member {index} has unsupported kind {kind!r}")
        filename = member["file"]
        if (not isinstance(filename, str) or not filename
                or Path(filename).name != filename):
            raise RuntimeError(f"member {index} has unsafe model filename {filename!r}")
        files.append(filename)
        try:
            weight = float(member["weight"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"member {index} weight is not numeric") from exc
        if not np.isfinite(weight) or weight <= 0:
            raise RuntimeError(f"member {index} weight must be finite and positive")
        if kind in {"binary", "multiclass"}:
            if "iterations" not in member:
                raise RuntimeError(f"member {index} is missing iterations")
            iterations = member["iterations"]
            if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
                raise RuntimeError(f"member {index} iterations must be a positive integer")
    duplicate_files = sorted({name for name in files if files.count(name) > 1})
    if duplicate_files:
        raise RuntimeError(f"model metadata references duplicate files: {duplicate_files}")

    shifts = value["shifts"]
    if not isinstance(shifts, dict) or set(shifts) != {"R", "F"}:
        raise RuntimeError("model metadata shifts must contain exactly R and F")
    for group, shift in shifts.items():
        try:
            numeric = float(shift)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"shift for {group} is not numeric") from exc
        if not np.isfinite(numeric):
            raise RuntimeError(f"shift for {group} must be finite")
    if not isinstance(value["entity_history"], dict) or not isinstance(value["encodings"], dict):
        raise RuntimeError("entity_history and encodings metadata must be dictionaries")
    return value


def validate_row_ids(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    for label, frame in (("test.csv", test), ("sample_submission.csv", submission)):
        if "row_id" not in frame.columns:
            raise RuntimeError(f"{label} is missing row_id")
        if frame["row_id"].isna().any():
            raise RuntimeError(f"{label} contains null row_id values")
        duplicates = frame.loc[frame["row_id"].duplicated(), "row_id"].head(5).tolist()
        if duplicates:
            raise RuntimeError(f"{label} contains duplicate row_id values: {duplicates}")
    test_ids = set(test["row_id"].tolist())
    submission_ids = set(submission["row_id"].tolist())
    if test_ids != submission_ids:
        only_test = list(test_ids - submission_ids)[:5]
        only_submission = list(submission_ids - test_ids)[:5]
        raise RuntimeError(
            "row_id set mismatch between test.csv and sample_submission.csv: "
            f"only_test={only_test}, only_submission={only_submission}")


def build_inference_matrix(frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    parts = [build_base(frame),
             entity_season_inference(frame, meta["entity_history"]),
             prev_game_counts(frame),
             transform_encodings(frame, meta["encodings"])]
    if meta["trackman"] is not None:
        parts.append(transform_trackman(frame, meta["trackman"]))
    matrix = pd.concat(parts, axis=1)
    if matrix.columns.duplicated().any():
        duplicates = matrix.columns[matrix.columns.duplicated()].tolist()
        raise RuntimeError(f"inference matrix has duplicate columns: {duplicates}")
    missing = [c for c in meta["columns"] if c not in matrix.columns]
    if missing:
        raise RuntimeError(f"inference code cannot reproduce model features: {missing}")
    return matrix[meta["columns"]].astype(np.float32)


def main() -> None:
    started = time.time()
    meta = validate_meta(pickle.loads((MODEL_DIR / "meta.pkl").read_bytes()))

    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig", low_memory=False)
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    validate_row_ids(test, submission)
    if "game_type" not in test.columns:
        raise RuntimeError("test.csv is missing game_type")
    unknown_groups = sorted(set(test["game_type"].dropna().tolist()) - set(meta["shifts"]))
    if test["game_type"].isna().any() or unknown_groups:
        raise RuntimeError(f"test.csv has unsupported game_type values: {unknown_groups}")

    matrix = build_inference_matrix(test, meta)
    print(f"features built in {time.time() - started:.0f}s", flush=True)

    total = np.zeros(len(test), dtype=np.float64)
    weight_sum = 0.0
    for index, member in enumerate(meta["members"]):
        kind = member["kind"]
        model_path = MODEL_DIR / member["file"]
        if not model_path.is_file():
            raise RuntimeError(f"member {index} model file is missing: {model_path}")
        if kind == "catboost":
            # imported lazily so requirements.txt only needs catboost when a
            # catboost member is actually packaged
            from catboost import CatBoostClassifier

            model = CatBoostClassifier()
            model.load_model(str(model_path))
            prediction = model.predict_proba(matrix)[:, 1]
        elif kind == "nn":
            bundle = pickle.loads(model_path.read_bytes())
            prediction = numpy_forward(bundle, nn_codes(test, matrix, bundle),
                                       nn_dense(matrix, bundle))
        elif kind in {"binary", "multiclass"}:
            booster = lgb.Booster(model_file=str(model_path))
            prediction = booster.predict(matrix, num_iteration=member["iterations"])
            if kind == "multiclass":
                if prediction.ndim != 2 or prediction.shape[1] != 4:
                    raise RuntimeError(
                        f"member {index} expected four-class output, got {prediction.shape}")
                prediction = prediction[:, 3]
        else:
            raise RuntimeError(f"member {index} has unsupported kind {kind!r}")
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != (len(test),):
            raise RuntimeError(
                f"member {index} returned shape {prediction.shape}, expected {(len(test),)}")
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"member {index} returned non-finite predictions")
        if (prediction < 0).any() or (prediction > 1).any():
            raise RuntimeError(f"member {index} returned probabilities outside [0, 1]")
        total += member["weight"] * prediction
        weight_sum += member["weight"]
    prediction = total / weight_sum

    shifts = meta["shifts"]
    game_type = test["game_type"].to_numpy()
    out = prediction.copy()
    for group, delta in shifts.items():
        mask = game_type == group
        out[mask] = apply_shift(out[mask], delta)
    out = np.clip(out, 1e-6, 1 - 1e-6)

    submission = submission[["row_id"]].merge(
        pd.DataFrame({"row_id": test["row_id"].to_numpy(), "control_success": out}),
        on="row_id", how="left", validate="one_to_one")
    if submission["control_success"].isna().any():
        raise RuntimeError("row_id mismatch between test.csv and sample_submission.csv")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {OUT_PATH} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

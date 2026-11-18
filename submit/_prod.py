"""LG Aimers 9 Phase 2 inference entry point.

Every prediction uses one evaluation row plus artifacts frozen from the
2019-2024 training data.  The code never aggregates, sorts, rolls, or target
encodes evaluation rows.  Trackman's raw CSV is not read during inference.
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6
DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
OUT_PATH = Path("./output/submission.csv")

DRIFT_COLS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
]

RAW_NUM = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

BASE_STATES = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]

PITCHER_RATE_COLUMNS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

BATTER_RATE_COLUMNS = [
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
]

CST_COMMAND_RATE_COLUMNS = (
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)
CST_MIX_RATE_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
CST_PROFILE_COLUMNS = [
    "cst_prev_reverse_rate",
    "cst_prev_middle_rate",
    "cst_prev_ball_rate",
    "cst_prev_strike_rate",
    "cst_reverse_delta_shr75",
    "cst_middle_delta_shr75",
    "cst_ball_delta_shr75",
    "cst_strike_delta_shr75",
    "cst_prev_fastball_rate",
    "cst_prev_breaking_rate",
    "cst_prev_offspeed_rate",
    "cst_fastball_offspeed_logratio_delta_shr75",
    "cst_breaking_offspeed_logratio_delta_shr75",
    "cst_command_drift_l2_shr75",
    "cst_pitchmix_hellinger_shr75",
]

TE_SPECS = [
    ("pte_game_type", ["pitcher_id", "game_type"]),
    ("pte_batter_hand", ["pitcher_id", "batter_hand"]),
    ("pte_count", ["pitcher_id", "balls_before", "strikes_before"]),
    ("bte_pitcher_hand", ["batter_id", "pitcher_hand"]),
    ("pair_te", ["pitcher_id", "batter_id"]),
]

RICH_TE_SPECS = [
    ("pte_base", ["pitcher_id", "base_state"], 180.0),
    (
        "pte_count_hand",
        ["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
        220.0,
    ),
    ("pte_batter_team", ["pitcher_id", "batter_team_id"], 220.0),
    ("pte_month", ["pitcher_id", "game_month"], 300.0),
    ("bte_count", ["batter_id", "balls_before", "strikes_before"], 300.0),
    (
        "bte_count_hand",
        ["batter_id", "balls_before", "strikes_before", "pitcher_hand"],
        360.0,
    ),
    ("bte_game_type", ["batter_id", "game_type"], 320.0),
    ("bte_pitcher_team", ["batter_id", "pitcher_team_id"], 360.0),
    (
        "pteam_count_hand",
        ["pitcher_team_id", "balls_before", "strikes_before", "batter_hand"],
        900.0,
    ),
    (
        "teams_count",
        ["pitcher_team_id", "batter_team_id", "balls_before", "strikes_before"],
        1200.0,
    ),
    (
        "state_te",
        [
            "game_type",
            "balls_before",
            "strikes_before",
            "outs_before",
            "base_state",
            "pitcher_hand",
            "batter_hand",
        ],
        1200.0,
    ),
]


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(denominator), np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _season_level(meta: dict, column: str, season: pd.Series) -> np.ndarray:
    table = meta["season_means"][column]
    default = float(table[str(meta["target_year"])])
    values = season.astype(int).astype(str).map(table).fillna(default)
    return values.to_numpy(np.float64)


def build_base_features(
    frame: pd.DataFrame,
    meta: dict,
    include_ids: bool,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Pure row-wise base feature transform."""
    output: dict[str, np.ndarray] = {}
    categorical_names: list[str] = []
    season = frame["season"]

    for column in RAW_NUM:
        output[column] = frame[column].to_numpy(np.float64)

    output["top_bottom_c"] = frame["top_bottom"].astype(str).eq("T").to_numpy(np.float64)
    output["game_type_c"] = frame["game_type"].astype(str).eq("R").to_numpy(np.float64)
    base_map = {value: index for index, value in enumerate(BASE_STATES)}
    output["base_state_c"] = (
        frame["base_state"].astype(str).map(base_map).fillna(-1).to_numpy(np.float64)
    )
    categorical_names.append("base_state_c")

    if include_ids:
        output["pitcher_c"] = (
            frame["pitcher_id"].astype(int).map(meta["pitcher_map"]).fillna(-1).to_numpy(np.float64)
        )
        output["batter_c"] = (
            frame["batter_id"].astype(int).map(meta["batter_map"]).fillna(-1).to_numpy(np.float64)
        )
        categorical_names.extend(["pitcher_c", "batter_c"])

    output["pteam_c"] = (
        frame["pitcher_team_id"].astype(int).map(meta["team_map"]).fillna(-1).to_numpy(np.float64)
    )
    output["bteam_c"] = (
        frame["batter_team_id"].astype(int).map(meta["team_map"]).fillna(-1).to_numpy(np.float64)
    )
    categorical_names.extend(["pteam_c", "bteam_c"])

    balls = frame["balls_before"].to_numpy(np.float64)
    strikes = frame["strikes_before"].to_numpy(np.float64)
    pitcher_n = frame["asof_pitcher_n"].to_numpy(np.float64)
    batter_n = frame["asof_batter_n"].to_numpy(np.float64)
    success = frame["asof_pitcher_success_rate"].to_numpy(np.float64)
    middle = frame["asof_pitcher_middle_rate"].to_numpy(np.float64)

    output["count_diff"] = balls - strikes
    output["count_c"] = balls * 3.0 + strikes
    categorical_names.append("count_c")
    output["is_3ball"] = (balls == 3).astype(np.float64)
    output["is_2strike"] = (strikes == 2).astype(np.float64)
    output["pitcher_ahead"] = (strikes > balls).astype(np.float64)
    output["batter_ahead"] = (balls > strikes).astype(np.float64)
    output["hand_match_c"] = (
        frame["pitcher_hand"].to_numpy(np.float64) * 2.0
        + frame["batter_hand"].to_numpy(np.float64)
    )
    categorical_names.append("hand_match_c")
    output["same_hand"] = (
        frame["pitcher_hand"].to_numpy() == frame["batter_hand"].to_numpy()
    ).astype(np.float64)

    score_difference = frame["score_diff_pitcher_team"].to_numpy(np.float64)
    output["abs_score_diff"] = np.abs(score_difference)
    output["close_game"] = (np.abs(score_difference) <= 2).astype(np.float64)
    inning = frame["inning"].to_numpy(np.float64)
    output["late"] = (inning >= 7).astype(np.float64)
    output["late_close"] = output["late"] * output["close_game"]
    output["blowout"] = (np.abs(score_difference) >= 5).astype(np.float64)
    output["log_li"] = np.log1p(frame["li"].to_numpy(np.float64))

    home_win_expectancy = frame["home_win_expectancy"].to_numpy(np.float64)
    output["pitcher_we"] = np.where(
        output["top_bottom_c"] > 0,
        home_win_expectancy,
        100.0 - home_win_expectancy,
    )
    output["we_margin"] = np.abs(home_win_expectancy - 50.0)
    output["is_pitcher_home"] = output["top_bottom_c"]
    output["log_pitcher_n"] = np.log1p(pitcher_n)
    output["log_batter_n"] = np.log1p(batter_n)
    output["pitcher_cold"] = (pitcher_n == 0).astype(np.float64)
    output["batter_cold"] = (batter_n == 0).astype(np.float64)
    output["prev_game_missing"] = (
        frame["asof_pitcher_prev1_game_success_rate"].isna().to_numpy(np.float64)
    )

    for window in (1, 3, 5):
        output[f"form{window}"] = (
            frame[f"asof_pitcher_prev{window}_game_success_rate"].to_numpy(np.float64)
            - success
        )
        output[f"form{window}_mid"] = (
            frame[f"asof_pitcher_prev{window}_game_middle_rate"].to_numpy(np.float64)
            - middle
        )
    output["form_slope"] = (
        frame["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
        - frame["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    )

    pitch_mix = frame[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        output["mix_entropy"] = -np.nansum(
            np.where(pitch_mix > 0, pitch_mix * np.log(pitch_mix + 1e-12), 0.0),
            axis=1,
        )
    output["nonfastball_rate"] = 1.0 - frame["asof_pitcher_fastball_rate"].to_numpy(np.float64)

    pitcher_level = _season_level(meta, "asof_pitcher_success_rate", season)
    batter_level = _season_level(meta, "asof_batter_success_rate", season)
    pitcher_probability = np.where(np.isfinite(success), success, pitcher_level)
    batter_probability = frame["asof_batter_success_rate"].to_numpy(np.float64)
    batter_probability = np.where(np.isfinite(batter_probability), batter_probability, batter_level)
    pitcher_probability = (pitcher_probability * pitcher_n + pitcher_level * 150.0) / (pitcher_n + 150.0)
    batter_probability = (batter_probability * batter_n + batter_level * 300.0) / (batter_n + 300.0)
    base_probability = 0.5 * (pitcher_level + batter_level)
    output["log5"] = logit(pitcher_probability) + logit(batter_probability) - logit(base_probability)
    output["pb_gap"] = logit(pitcher_probability) - logit(batter_probability)

    pitcher_middle_level = _season_level(meta, "asof_pitcher_middle_rate", season)
    batter_middle_level = _season_level(meta, "asof_batter_middle_rate", season)
    pitcher_middle = np.where(np.isfinite(middle), middle, pitcher_middle_level)
    batter_middle = frame["asof_batter_middle_rate"].to_numpy(np.float64)
    batter_middle = np.where(np.isfinite(batter_middle), batter_middle, batter_middle_level)
    pitcher_middle = (pitcher_middle * pitcher_n + pitcher_middle_level * 150.0) / (pitcher_n + 150.0)
    batter_middle = (batter_middle * batter_n + batter_middle_level * 300.0) / (batter_n + 300.0)
    output["log5_mid"] = (
        logit(pitcher_middle)
        + logit(batter_middle)
        - logit(0.5 * (pitcher_middle_level + batter_middle_level))
    )

    names = list(output)
    matrix = np.column_stack([output[name] for name in names]).astype(np.float32)
    categorical_indices = [names.index(name) for name in categorical_names]
    return matrix, names, categorical_indices


def _row_prior(frame: pd.DataFrame, meta: dict) -> np.ndarray:
    priors = meta["priors"]
    return (
        frame["game_type"].astype(str).map(priors).fillna(priors["__overall__"]).to_numpy(np.float64)
    )


def _calibrate_component(
    prediction: np.ndarray,
    game_type: np.ndarray,
    offsets: dict[str, float],
) -> np.ndarray:
    row_offsets = np.full(
        len(prediction), float(offsets["__overall__"]), dtype=np.float64
    )
    for group in ("R", "F"):
        row_offsets[game_type == group] = float(offsets[group])
    return sigmoid(logit(prediction) + row_offsets)


def _entity_season_features(
    frame: pd.DataFrame,
    meta: dict,
    entity_column: str,
    n_column: str,
    rate_columns: list[str],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    history = meta["entity_history"][prefix]
    entity_ids = frame[entity_column].astype(int)
    start_n = entity_ids.map(history["n"]).fillna(0).to_numpy(np.float64)
    start_success = entity_ids.map(history["success"]).fillna(0).to_numpy(np.float64)
    prior_rate = entity_ids.map(history["prior_rate"]).to_numpy(np.float64)
    previous_n = entity_ids.map(history["previous_n"]).fillna(0).to_numpy(np.float64)
    previous_rate = entity_ids.map(history["previous_rate"]).to_numpy(np.float64)
    asof_n = frame[n_column].to_numpy(np.float64)
    asof_success = np.rint(
        np.nan_to_num(frame[rate_columns[0]].to_numpy(np.float64) * asof_n, nan=0.0)
    )
    season_n = np.maximum(asof_n - start_n, 0.0)
    season_success = np.maximum(asof_success - start_success, 0.0)
    season_rate = safe_rate(season_success, season_n)
    default_prior = _row_prior(frame, meta)
    shrink_prior = np.where(
        np.isfinite(previous_rate),
        previous_rate,
        np.where(np.isfinite(prior_rate), prior_rate, default_prior),
    )

    output: dict[str, np.ndarray] = {
        f"{prefix}_season_n": season_n,
        f"{prefix}_season_success_count": season_success,
        f"{prefix}_season_success_rate": season_rate,
        f"{prefix}_season_success_shr25": (season_success + 25.0 * shrink_prior) / (season_n + 25.0),
        f"{prefix}_season_success_shr75": (season_success + 75.0 * shrink_prior) / (season_n + 75.0),
        f"{prefix}_prev_season_n": previous_n,
        f"{prefix}_prev_season_success_rate": previous_rate,
        f"{prefix}_prior_seasons_success_rate": prior_rate,
        f"{prefix}_season_minus_prev": season_rate - previous_rate,
        f"{prefix}_season_progress": season_n / (previous_n + 50.0),
    }

    category_history = history["category_counts"]
    for column in rate_columns[1:]:
        baseline = entity_ids.map(category_history[column]).fillna(0).to_numpy(np.float64)
        asof_category = np.rint(
            np.nan_to_num(frame[column].to_numpy(np.float64) * asof_n, nan=0.0)
        )
        season_category = np.maximum(asof_category - baseline, 0.0)
        output[f"{prefix}_season_{column.removeprefix('asof_')}"] = safe_rate(
            season_category, season_n
        )

    names = list(output)
    matrix = np.column_stack([output[name] for name in names]).astype(np.float32)
    return matrix, names


def build_season_features(frame: pd.DataFrame, meta: dict) -> tuple[np.ndarray, list[str]]:
    pitcher_matrix, pitcher_names = _entity_season_features(
        frame,
        meta,
        "pitcher_id",
        "asof_pitcher_n",
        PITCHER_RATE_COLUMNS,
        "p",
    )
    batter_matrix, batter_names = _entity_season_features(
        frame,
        meta,
        "batter_id",
        "asof_batter_n",
        BATTER_RATE_COLUMNS,
        "b",
    )
    return (
        np.column_stack([pitcher_matrix, batter_matrix]).astype(np.float32),
        pitcher_names + batter_names,
    )


def build_te_features(frame: pd.DataFrame, meta: dict) -> tuple[np.ndarray, list[str]]:
    pieces: list[np.ndarray] = []
    names: list[str] = []
    prior = float(meta["te_prior"])
    for name, keys in TE_SPECS:
        specification = meta["target_encodings"][name]
        key_values = pd.Series(list(zip(*(frame[key].tolist() for key in keys))))
        numerators = key_values.map(specification["sum"]).to_numpy(np.float64)
        denominators = key_values.map(specification["count"]).to_numpy(np.float64)
        alpha = float(specification["alpha"])
        encoded = (numerators + alpha * prior) / (denominators + alpha)
        pieces.extend([encoded.astype(np.float32), np.log1p(denominators).astype(np.float32)])
        names.extend([name, f"{name}_logn"])
    return np.column_stack(pieces).astype(np.float32), names


def build_rich_decay_features(
    frame: pd.DataFrame,
    meta: dict,
) -> tuple[np.ndarray, list[str]]:
    """Look up target encodings frozen from 2019-2024 with a two-year half-life."""
    pieces: list[np.ndarray] = []
    names: list[str] = []
    prior = float(meta["rich_te_prior"])
    for name, keys, alpha in RICH_TE_SPECS:
        specification = meta["rich_target_encodings"][name]
        key_values = pd.Series(list(zip(*(frame[key].tolist() for key in keys))))
        numerators = key_values.map(specification["weighted_sum"]).to_numpy(np.float64)
        denominators = key_values.map(specification["weighted_count"]).to_numpy(np.float64)
        encoded = (numerators + alpha * prior) / (denominators + alpha)
        pieces.extend(
            [
                encoded.astype(np.float32),
                np.log1p(denominators).astype(np.float32),
            ]
        )
        names.extend([f"{name}_decay", f"{name}_decay_logn"])
    return np.column_stack(pieces).astype(np.float32), names


def build_trackman_features(frame: pd.DataFrame, meta: dict) -> tuple[np.ndarray, list[str]]:
    trackman = meta["trackman"]
    pitcher_ids = frame["pitcher_id"].astype(int)
    mapped_ids = pitcher_ids.map(trackman["pitcher_to_trackman"])
    pieces: list[np.ndarray] = []
    names: list[str] = []
    for column in trackman["mapping_columns"]:
        pieces.append(pitcher_ids.map(trackman["mapping_values"][column]).to_numpy(np.float32))
        names.append(column)
    pieces.append(mapped_ids.notna().to_numpy(np.float32))
    names.append("tm_map_available")
    for column in trackman["profile_columns"]:
        pieces.append(mapped_ids.map(trackman["profile_values"][column]).to_numpy(np.float32))
        names.append(column)
    return np.column_stack(pieces).astype(np.float32), names


def build_batter_trackman_features(
    frame: pd.DataFrame, meta: dict
) -> tuple[np.ndarray, list[str]]:
    """Frozen physical profile of the pitches this batter has faced.

    Row-wise: the batter id on the row selects a frozen lookup built from
    TrackMan through 2024.  No evaluation row is consulted for another.
    """
    spec = meta["trackman_batter"]
    batter_ids = frame["batter_id"].astype(int)
    mapped = batter_ids.map(spec["batter_to_trackman"])
    pieces = [mapped.notna().to_numpy(np.float32)]
    names = ["tmb_map_available"]
    for column in spec["profile_columns"]:
        pieces.append(mapped.map(spec["profile_values"][column]).to_numpy(np.float32))
        names.append(column)
    return np.column_stack(pieces).astype(np.float32), names


def build_trackman_temporal_features(
    frame: pd.DataFrame, meta: dict
) -> tuple[np.ndarray, list[str]]:
    """Attach the frozen TMT-1 pitcher trajectory for each row independently."""
    spec = meta["trackman_temporal"]
    if spec.get("schema_version") != 1 or int(spec.get("cutoff", -1)) < 2019:
        raise RuntimeError("unsupported TMT-1 metadata contract")
    pitcher_ids = frame["pitcher_id"].astype(int)
    mapped = pitcher_ids.map(meta["trackman"]["pitcher_to_trackman"])
    names = list(spec["profile_columns"])
    if len(names) != 31 or len(names) != len(set(names)):
        raise RuntimeError("TMT-1 profile layout is invalid")
    pieces = [
        mapped.map(spec["profile_values"][column]).to_numpy(np.float32)
        for column in names
    ]
    return np.column_stack(pieces).astype(np.float32), names


def _cst_rate_from_cumulative(
    frame: pd.DataFrame,
    pitcher_ids: pd.Series,
    start_n: np.ndarray,
    current_n: np.ndarray,
    n_column: str,
    rate_column: str,
    start_counts: dict,
) -> np.ndarray:
    denominator = frame[n_column].to_numpy(np.float64)
    rate = frame[rate_column].to_numpy(np.float64)
    cumulative = np.where(
        np.isfinite(rate), np.rint(rate * denominator), np.nan
    )
    baseline = pitcher_ids.map(start_counts).to_numpy(np.float64)
    numerator = cumulative - baseline
    result = np.full(len(frame), np.nan, dtype=np.float64)
    usable = (
        np.isfinite(start_n)
        & np.isfinite(current_n)
        & (current_n > 0)
        & np.isfinite(numerator)
        & (numerator >= 0)
    )
    result[usable] = numerator[usable] / current_n[usable]
    return result


def build_cst_features(
    frame: pd.DataFrame, meta: dict
) -> tuple[np.ndarray, list[str]]:
    """Attach the row-wise CST-1 current-versus-previous command state."""
    spec = meta["command_state_transition"]
    if spec.get("schema_version") != 1:
        raise RuntimeError("unsupported CST-1 metadata contract")
    names = list(spec.get("profile_columns", []))
    if names != CST_PROFILE_COLUMNS or len(names) != len(set(names)):
        raise RuntimeError("CST-1 profile layout is invalid")
    if tuple(spec.get("command_rate_columns", [])) != CST_COMMAND_RATE_COLUMNS:
        raise RuntimeError("CST-1 command column contract changed")
    if tuple(spec.get("mix_rate_columns", [])) != CST_MIX_RATE_COLUMNS:
        raise RuntimeError("CST-1 pitch-mix column contract changed")
    shrinkage = float(spec.get("shrinkage_count", np.nan))
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise RuntimeError("CST-1 shrinkage count is invalid")

    pitcher_ids = frame["pitcher_id"].astype(int)
    current_start = spec["current_start"]
    previous = spec["previous"]
    start_command_n = pitcher_ids.map(current_start["command_n"]).to_numpy(
        np.float64
    )
    row_command_n = frame["asof_pitcher_n"].to_numpy(np.float64)
    current_command_n = row_command_n - start_command_n
    previous_command_n = pitcher_ids.map(previous["command_n"]).to_numpy(
        np.float64
    )
    command_weight = (
        current_command_n / (current_command_n + shrinkage)
    ) * (
        previous_command_n / (previous_command_n + shrinkage)
    )
    command_weight[
        ~(
            np.isfinite(current_command_n)
            & (current_command_n > 0)
            & np.isfinite(previous_command_n)
            & (previous_command_n > 0)
        )
    ] = np.nan

    command_previous: list[np.ndarray] = []
    command_delta: list[np.ndarray] = []
    for column in CST_COMMAND_RATE_COLUMNS:
        prior = pitcher_ids.map(previous["command_rates"][column]).to_numpy(
            np.float64
        )
        current = _cst_rate_from_cumulative(
            frame,
            pitcher_ids,
            start_command_n,
            current_command_n,
            "asof_pitcher_n",
            column,
            current_start["command_counts"][column],
        )
        command_previous.append(prior)
        command_delta.append((current - prior) * command_weight)

    start_mix_n = pitcher_ids.map(current_start["mix_n"]).to_numpy(np.float64)
    row_mix_n = frame["asof_pitcher_pitchmix_n"].to_numpy(np.float64)
    current_mix_n = row_mix_n - start_mix_n
    previous_mix_n = pitcher_ids.map(previous["mix_n"]).to_numpy(np.float64)
    mix_weight = (current_mix_n / (current_mix_n + shrinkage)) * (
        previous_mix_n / (previous_mix_n + shrinkage)
    )
    mix_weight[
        ~(
            np.isfinite(current_mix_n)
            & (current_mix_n > 0)
            & np.isfinite(previous_mix_n)
            & (previous_mix_n > 0)
        )
    ] = np.nan

    mix_previous: list[np.ndarray] = []
    mix_current: list[np.ndarray] = []
    for column in CST_MIX_RATE_COLUMNS:
        prior = pitcher_ids.map(previous["mix_rates"][column]).to_numpy(
            np.float64
        )
        current = _cst_rate_from_cumulative(
            frame,
            pitcher_ids,
            start_mix_n,
            current_mix_n,
            "asof_pitcher_pitchmix_n",
            column,
            current_start["mix_counts"][column],
        )
        mix_previous.append(prior)
        mix_current.append(current)

    previous_mix = np.column_stack(mix_previous)
    current_mix = np.column_stack(mix_current)
    previous_total = previous_mix.sum(axis=1)
    current_total = current_mix.sum(axis=1)
    mix_usable = (
        np.isfinite(previous_mix).all(axis=1)
        & np.isfinite(current_mix).all(axis=1)
        & (previous_total > 0)
        & (current_total > 0)
    )
    previous_normalized = np.full_like(previous_mix, np.nan)
    current_normalized = np.full_like(current_mix, np.nan)
    previous_normalized[mix_usable] = (
        previous_mix[mix_usable] / previous_total[mix_usable, None]
    )
    current_normalized[mix_usable] = (
        current_mix[mix_usable] / current_total[mix_usable, None]
    )
    ratio_eps = 1e-4
    fastball_offspeed_delta = (
        np.log(
            (current_normalized[:, 0] + ratio_eps)
            / (current_normalized[:, 2] + ratio_eps)
        )
        - np.log(
            (previous_normalized[:, 0] + ratio_eps)
            / (previous_normalized[:, 2] + ratio_eps)
        )
    ) * mix_weight
    breaking_offspeed_delta = (
        np.log(
            (current_normalized[:, 1] + ratio_eps)
            / (current_normalized[:, 2] + ratio_eps)
        )
        - np.log(
            (previous_normalized[:, 1] + ratio_eps)
            / (previous_normalized[:, 2] + ratio_eps)
        )
    ) * mix_weight
    command_drift = np.sqrt(
        np.mean(np.column_stack(command_delta) ** 2, axis=1)
    )
    pitchmix_hellinger = np.sqrt(
        0.5
        * np.sum(
            (
                np.sqrt(np.clip(current_normalized, 0.0, 1.0))
                - np.sqrt(np.clip(previous_normalized, 0.0, 1.0))
            )
            ** 2,
            axis=1,
        )
    ) * mix_weight

    output = {
        "cst_prev_reverse_rate": command_previous[0],
        "cst_prev_middle_rate": command_previous[1],
        "cst_prev_ball_rate": command_previous[2],
        "cst_prev_strike_rate": command_previous[3],
        "cst_reverse_delta_shr75": command_delta[0],
        "cst_middle_delta_shr75": command_delta[1],
        "cst_ball_delta_shr75": command_delta[2],
        "cst_strike_delta_shr75": command_delta[3],
        "cst_prev_fastball_rate": previous_normalized[:, 0],
        "cst_prev_breaking_rate": previous_normalized[:, 1],
        "cst_prev_offspeed_rate": previous_normalized[:, 2],
        "cst_fastball_offspeed_logratio_delta_shr75": fastball_offspeed_delta,
        "cst_breaking_offspeed_logratio_delta_shr75": breaking_offspeed_delta,
        "cst_command_drift_l2_shr75": command_drift,
        "cst_pitchmix_hellinger_shr75": pitchmix_hellinger,
    }
    if list(output) != names:
        raise RuntimeError("CST-1 output order changed")
    return (
        np.column_stack([output[name] for name in names]).astype(np.float32),
        names,
    )


def _mask_trackman_confidence_in_place(
    matrix: np.ndarray,
    names: list[str],
    meta: dict,
) -> None:
    gate = meta["trackman_confidence_gate"]
    expected = {"score_lt", "margin_gt", "overlap_ge"}
    if set(gate) != expected:
        raise RuntimeError(
            f"invalid TrackMan confidence gate keys: {sorted(gate)}"
        )
    score_limit = float(gate["score_lt"])
    margin_limit = float(gate["margin_gt"])
    overlap_limit = float(gate["overlap_ge"])
    if not np.isfinite([score_limit, margin_limit, overlap_limit]).all():
        raise RuntimeError("TrackMan confidence gate contains non-finite values")

    required = [
        "tm_map_score",
        "tm_map_margin",
        "tm_map_overlap_years",
        "tm_map_available",
        *meta["trackman"]["profile_columns"],
        *meta.get("trackman_temporal", {}).get("profile_columns", []),
    ]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"TrackMan confidence features are missing: {missing}")
    score = matrix[:, names.index("tm_map_score")]
    margin = matrix[:, names.index("tm_map_margin")]
    overlap = matrix[:, names.index("tm_map_overlap_years")]
    retained = (
        (score < score_limit)
        & (margin > margin_limit)
        & (overlap >= overlap_limit)
    )
    rejected_rows = np.flatnonzero(~retained)
    profile_indices = np.asarray(
        [
            names.index(name)
            for name in (
                list(meta["trackman"]["profile_columns"])
                + list(meta.get("trackman_temporal", {}).get("profile_columns", []))
            )
        ],
        dtype=np.int64,
    )
    if rejected_rows.size:
        matrix[rejected_rows[:, None], profile_indices] = np.nan
        matrix[rejected_rows, names.index("tm_map_available")] = 0.0


def _catboost_frame(matrix: np.ndarray, names: list[str], cat_indices: list[int]) -> pd.DataFrame:
    result = pd.DataFrame(matrix, columns=names)
    for index in cat_indices:
        column = names[index]
        result[column] = result[column].fillna(-1).astype(np.int32)
    return result


def predict(frame: pd.DataFrame, meta: dict,
            capture: dict | None = None) -> np.ndarray:
    started = time.perf_counter()
    target_year = int(meta["target_year"])
    season = frame["season"]
    if season.isna().any() or not season.astype(int).eq(target_year).all():
        raise ValueError(f"every evaluation row must have season={target_year}")

    season_matrix, season_names = build_season_features(frame, meta)
    te_matrix, te_names = build_te_features(frame, meta)
    rich_decay_matrix, rich_decay_names = build_rich_decay_features(frame, meta)

    cat_base, cat_base_names, cat_indices = build_base_features(frame, meta, include_ids=True)
    cat_matrix = np.column_stack([cat_base, season_matrix, te_matrix]).astype(np.float32)
    cat_names = cat_base_names + season_names + te_names
    if cat_names != meta["cat_feature_names"] or cat_indices != meta["cat_indices"]:
        raise RuntimeError("CatBoost feature layout does not match training")
    cat_frame = _catboost_frame(cat_matrix, cat_names, cat_indices)
    cat_pool = Pool(cat_frame, cat_features=cat_indices, feature_names=cat_names)
    cat_decay_matrix = np.column_stack([cat_matrix, rich_decay_matrix]).astype(
        np.float32
    )
    cat_decay_names = cat_names + rich_decay_names
    if cat_decay_names != meta["cat_decay_feature_names"]:
        raise RuntimeError("Decayed CatBoost feature layout does not match training")
    cat_decay_frame = _catboost_frame(
        cat_decay_matrix, cat_decay_names, cat_indices
    )
    cat_decay_pool = Pool(
        cat_decay_frame,
        cat_features=cat_indices,
        feature_names=cat_decay_names,
    )
    # Exact-equivalent CatBoost model-parallel path.  The models are
    # independent; only their weighted predictions are combined.  Each C++
    # prediction is single-threaded and a small number of members run in
    # parallel, preventing nested thread oversubscription.  Accumulation below
    # remains in the original metadata order.
    from concurrent.futures import ThreadPoolExecutor

    cat_components = [
        component
        for component in meta["stack_coefficients"]
        if component.startswith("catboost")
    ]
    cat_predictions = {
        component: np.zeros(len(frame), dtype=np.float64)
        for component in cat_components
    }
    cat_weights = {component: 0.0 for component in cat_components}
    cat_members = list(meta["catboost_models"])
    requested_workers = int(os.environ.get("CATBOOST_MODEL_WORKERS", "4"))
    if requested_workers < 1:
        raise RuntimeError("CATBOOST_MODEL_WORKERS must be >= 1")
    cat_workers = max(1, min(requested_workers, int(os.cpu_count() or 1)))

    def _predict_catboost_member(member):
        component = str(member["component"])
        model_type = str(member["model_type"])
        model = CatBoostRegressor() if model_type == "regressor" else CatBoostClassifier()
        model.load_model(str(MODEL_DIR / member["file"]))
        pool = cat_decay_pool if member["feature_set"] == "decay" else cat_pool
        if model_type == "regressor":
            prediction = np.clip(
                model.predict(pool, thread_count=1), 1e-5, 1.0 - 1e-5
            )
        else:
            prediction = model.predict_proba(pool, thread_count=1)[:, 1]
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise RuntimeError(
                f"CatBoost member {member['file']} produced invalid output: {prediction.shape}"
            )
        return component, float(member["weight"]), prediction

    with ThreadPoolExecutor(max_workers=cat_workers) as executor:
        for start in range(0, len(cat_members), cat_workers):
            batch = cat_members[start:start + cat_workers]
            futures = [executor.submit(_predict_catboost_member, member) for member in batch]
            # Resolve and accumulate in metadata order, never completion order.
            results = [future.result() for future in futures]
            for component, member_weight, member_prediction in results:
                cat_predictions[component] += member_weight * member_prediction
                cat_weights[component] += member_weight

    for component in cat_predictions:
        cat_predictions[component] /= cat_weights[component]
    del cat_pool, cat_decay_pool, cat_frame, cat_decay_frame, cat_matrix, cat_decay_matrix, cat_base

    lgb_base, lgb_base_names, lgb_cat_indices = build_base_features(
        frame, meta, include_ids=False
    )
    trackman_matrix, trackman_names = build_trackman_features(frame, meta)
    batter_tm_matrix, batter_tm_names = build_batter_trackman_features(frame, meta)
    temporal_tm_matrix, temporal_tm_names = build_trackman_temporal_features(
        frame, meta
    )
    cst_matrix, cst_names = build_cst_features(frame, meta)
    lgb_matrix = np.column_stack(
        [
            lgb_base,
            season_matrix,
            te_matrix,
            trackman_matrix,
            batter_tm_matrix,
            temporal_tm_matrix,
            cst_matrix,
        ]
    ).astype(np.float32)
    lgb_names = (
        lgb_base_names
        + season_names
        + te_names
        + trackman_names
        + batter_tm_names
        + temporal_tm_names
        + cst_names
    )
    if lgb_names != meta["lgb_feature_names"] or lgb_cat_indices != meta["lgb_cat_indices"]:
        raise RuntimeError("LightGBM feature layout does not match training")
    lgb_decay_matrix = np.column_stack([lgb_matrix, rich_decay_matrix]).astype(
        np.float32
    )
    lgb_decay_names = lgb_names + rich_decay_names
    # Hand the shipped decay matrix to the caller so a count-segment
    # specialist can reuse it instead of paying the feature build twice.
    if capture is not None:
        capture["lgb_decay_matrix"] = lgb_decay_matrix
    if lgb_decay_names != meta["lgb_decay_feature_names"]:
        raise RuntimeError("Decayed LightGBM feature layout does not match training")

    # Optional Futures-league specialist.  It was trained on the full unmasked
    # LightGBM matrix, so it has to predict before the in-place TrackMan
    # confidence gate below rewrites lgb_matrix.
    specialist_prediction = None
    specialist = meta.get("league_specialist")
    if specialist is not None:
        specialist_path = MODEL_DIR / str(specialist["file"])
        if not specialist_path.is_file():
            raise RuntimeError(f"league specialist model is missing: {specialist_path}")
        digest = hashlib.sha256(specialist_path.read_bytes()).hexdigest()
        if digest != str(specialist["sha256"]):
            raise RuntimeError(
                f"league specialist hash mismatch: expected {specialist['sha256']} "
                f"actual {digest}"
            )
        specialist_model = lgb.Booster(model_file=str(specialist_path))
        if specialist_model.num_feature() != len(lgb_names):
            raise RuntimeError(
                f"league specialist expects {specialist_model.num_feature()} features, "
                f"the production matrix has {len(lgb_names)}"
            )
        specialist_prediction = np.asarray(
            specialist_model.predict(
                lgb_matrix, num_iteration=int(specialist["iterations"])
            ),
            dtype=np.float64,
        )
        if not np.isfinite(specialist_prediction).all():
            raise RuntimeError("league specialist produced non-finite output")
        del specialist_model
    lgb_components = [
        component
        for component in meta["stack_coefficients"]
        if component.startswith("lgb")
    ]
    lgb_predictions: dict[str, np.ndarray] = {
        component: np.zeros(len(frame), dtype=np.float64)
        for component in lgb_components
    }
    lgb_weights = {component: 0.0 for component in lgb_components}
    confidence_gate_applied = False
    for member in meta["lightgbm_models"]:
        model = lgb.Booster(model_file=str(MODEL_DIR / member["file"]))
        component = str(member["component"])
        member_weight = float(member["weight"])
        if component == "lgb15":
            if confidence_gate_applied:
                raise RuntimeError("LGB15 must run before the in-place TrackMan gate")
            feature_matrix = lgb_matrix
        elif component == "lgb31":
            if not confidence_gate_applied:
                _mask_trackman_confidence_in_place(lgb_matrix, lgb_names, meta)
                confidence_gate_applied = True
            feature_matrix = lgb_matrix
        elif component == "lgb_decay":
            feature_matrix = lgb_decay_matrix
        else:
            raise RuntimeError(f"unknown LightGBM component: {component}")
        lgb_predictions[component] += member_weight * model.predict(feature_matrix)
        lgb_weights[component] += member_weight
    if not confidence_gate_applied:
        raise RuntimeError("base LGB31 TrackMan confidence gate was not applied")
    for component in lgb_predictions:
        lgb_predictions[component] /= lgb_weights[component]

    game_type = frame["game_type"].astype(str).to_numpy()
    all_predictions = {**cat_predictions, **lgb_predictions}
    calibrated_components = {
        component: _calibrate_component(
            component_prediction,
            game_type,
            meta["component_calibration_offsets"][component],
        )
        for component, component_prediction in all_predictions.items()
    }
    # Route the LightGBM slot through the specialist on its own league.  Both
    # slot members are overwritten on those rows, so the slot's whole
    # coefficient lands on the specialist ("replace") or on a three-way mean
    # that keeps the generalists ("blend").
    if specialist_prediction is not None:
        league_mask = game_type == str(specialist["league"])
        mode = str(specialist["mode"])
        slot_members = [
            name for name in ("lgb31", "lgb_decay") if name in calibrated_components
        ]
        if mode == "replace":
            replacement = specialist_prediction[league_mask]
        elif mode == "blend":
            replacement = np.mean(
                [calibrated_components[name][league_mask] for name in slot_members]
                + [specialist_prediction[league_mask]],
                axis=0,
            )
        else:
            raise RuntimeError(f"unsupported league specialist mode: {mode!r}")
        for name in slot_members:
            calibrated_components[name] = calibrated_components[name].copy()
            calibrated_components[name][league_mask] = replacement

    row_prior = _row_prior(frame, meta)
    prediction = row_prior.copy()
    for component, coefficient in meta["stack_coefficients"].items():
        prediction += float(coefficient) * (
            calibrated_components[component] - row_prior
        )
    prediction = np.clip(prediction, EPS, 1.0 - EPS)
    if not np.isfinite(prediction).all():
        raise RuntimeError("model produced non-finite probabilities")
    return prediction


def _validate_inputs(test: pd.DataFrame, submission: pd.DataFrame, meta: dict) -> None:
    missing_columns = sorted(set(meta["required_columns"]) - set(test.columns))
    if missing_columns:
        raise ValueError(f"test.csv is missing columns: {missing_columns}")
    if list(submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission.csv columns must be {[ID_COL, TARGET_COL]}, "
            f"received {list(submission.columns)}"
        )
    if test[ID_COL].isna().any() or submission[ID_COL].isna().any():
        raise ValueError("row_id contains null values")
    if not test[ID_COL].is_unique or not submission[ID_COL].is_unique:
        raise ValueError("row_id must be unique in both input files")
    test_ids = pd.Index(test[ID_COL])
    submission_ids = pd.Index(submission[ID_COL])
    if len(test_ids) != len(submission_ids) or set(test_ids) != set(submission_ids):
        missing = submission_ids.difference(test_ids).tolist()[:5]
        extra = test_ids.difference(submission_ids).tolist()[:5]
        raise ValueError(
            "test/sample_submission row_id mismatch: "
            f"test={len(test_ids)} submission={len(submission_ids)} "
            f"missing_examples={missing} extra_examples={extra}"
        )


def main() -> None:
    total_started = time.perf_counter()
    with (MODEL_DIR / "meta.pkl").open("rb") as file:
        meta = pickle.load(file)

    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig", low_memory=False)
    submission = pd.read_csv(
        DATA_DIR / "sample_submission.csv", encoding="utf-8-sig", low_memory=False
    )
    _validate_inputs(test, submission, meta)

    probabilities = predict(test, meta)
    indexed_prediction = pd.Series(probabilities, index=test[ID_COL].to_numpy())
    ordered = indexed_prediction.reindex(submission[ID_COL]).to_numpy(np.float64)
    if not np.isfinite(ordered).all():
        raise RuntimeError("row_id alignment produced missing predictions")
    submission[TARGET_COL] = ordered

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

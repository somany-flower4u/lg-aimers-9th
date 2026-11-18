"""Train and package the high-accuracy LG Aimers Phase 2 submission.

This file is not included in submit.zip.  It creates frozen models and metadata
under submit/model; submit/script.py performs evaluation-time inference.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "offer/data/train.csv"
TRACKMAN_PATH = ROOT / "offer/data/trackman_history.csv"
MODEL_DIR = ROOT / "submit/model"
TARGET_YEAR = 2025
VALID_YEAR = 2024
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
MEASURES = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
TMT_GROUPS = tuple(PITCH_GROUPS[:3])
TMT_EVIDENCE_COLUMNS = (
    "tmt_last_log_n",
    "tmt_prior_log_n",
    "tmt_profile_age",
    "tmt_two_season_available",
)
TMT_PROFILE_COLUMNS = list(TMT_EVIDENCE_COLUMNS) + [
    column
    for group in TMT_GROUPS
    for column in (
        f"tmt_{group}_share_delta",
        *(f"tmt_{group}_{measure}_z_delta" for measure in MEASURES),
    )
]
CST_SHRINKAGE_N = 75.0
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
TE_DEFINITIONS = [
    ("pte_game_type", ["pitcher_id", "game_type"], 100.0),
    ("pte_batter_hand", ["pitcher_id", "batter_hand"], 120.0),
    ("pte_count", ["pitcher_id", "balls_before", "strikes_before"], 180.0),
    ("bte_pitcher_hand", ["batter_id", "pitcher_hand"], 250.0),
    ("pair_te", ["pitcher_id", "batter_id"], 80.0),
]
FINAL_ROUNDS = {
    "catboost": 126,
    "catboost_decay_classifier": 126,
    "catboost_decay_regressor": 170,
    "lgb15": 280,
    "lgb31": 184,
    "lgb_decay": 220,
}
# Seed averaging is the only lever here whose effect is an identity rather than
# an estimate: Brier is squared error, so averaging k draws of the same
# estimator removes exactly (1 - 1/k) of the per-seed variance.  The first two
# seeds of each component were shipped; the rest extend the same average.
# Measured seed spread put the remaining headroom almost entirely in the three
# CatBoost components, so lgb15 and lgb31 are deliberately left alone.
# The first five of each tuple are the seeds that scored 1093.114.  The rest
# are a contiguous range rather than chosen values, so no seed can have been
# picked for its result.
CATBOOST_SEEDS = (426, 990, 137, 251, 863) + tuple(range(1001, 1011))
LGB_DECAY_SEEDS = (303, 991, 145, 277, 419) + tuple(range(2001, 2011))
STACK_COEFFICIENTS = {
    "catboost": 0.24437975,
    "catboost_decay_classifier": 0.24437975,
    "catboost_decay_regressor": 0.24437975,
    "lgb15": 0.09498236,
    "lgb31": 0.112113555,
    "lgb_decay": 0.112113555,
}
OOF_VALIDATION = {
    "year": 2024,
    "brier": 0.2475102039747933,
    "brier_skill": 919.3991583070659,
    "auc": 0.5538765196419613,
    "prediction_mean": 0.4872751274661518,
    "stack_training": (
        "2022-2024 rolling OOF, equal family averaging, strict same-row "
        "Trackman endpoints"
    ),
}


def load_solution_module():
    # submit/script.py composes the production stack with an auxiliary member.
    # Training and rolling-OOF experiments need the underlying production
    # feature implementation, which is kept separately for row-independent
    # inference reuse.
    path = ROOT / "submit/_prod.py"
    specification = importlib.util.spec_from_file_location("submission_solution", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import production solution from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def forecast_linear(series: pd.Series, target_year: int) -> float:
    clean = series.dropna().astype(float)
    if len(clean) == 0:
        return 0.5
    if len(clean) == 1:
        return float(clean.iloc[-1])
    coefficients = np.polyfit(clean.index.to_numpy(float), clean.to_numpy(), 1)
    return float(np.polyval(coefficients, target_year))


def forecast_blend(series: pd.Series, target_year: int, trend_weight: float) -> float:
    last = float(series.dropna().iloc[-1])
    trend = forecast_linear(series, target_year)
    return float(last + trend_weight * (trend - last))


def forecast_futures_prior(series: pd.Series, target_year: int) -> float:
    """Shock-aware one-step forecast for the volatile Futures league.

    A large final-year level break is carried forward once instead of being
    diluted by the older regime.  Once two observations exist in the new
    regime, only one quarter of the latest annual change is extrapolated.
    """
    clean = series.dropna().astype(float).sort_index()
    if len(clean) < 2:
        return float(clean.iloc[-1]) if len(clean) else 0.5
    last = float(clean.iloc[-1])
    previous = float(clean.iloc[-2])
    if abs(last - previous) >= 0.10:
        return last
    return float(np.clip(last + 0.25 * (last - previous), 0.05, 0.95))


def make_priors(full: pd.DataFrame) -> dict[str, float]:
    overall = full.groupby("season", observed=True).control_success.mean()
    regular = (
        full.loc[full.game_type.eq("R")]
        .groupby("season", observed=True)
        .control_success.mean()
    )
    futures = (
        full.loc[full.game_type.eq("F")]
        .groupby("season", observed=True)
        .control_success.mean()
    )
    return {
        "__overall__": forecast_blend(overall, TARGET_YEAR, 0.67),
        "R": forecast_blend(regular, TARGET_YEAR, 0.37),
        "F": forecast_futures_prior(futures, TARGET_YEAR),
    }


def make_base_meta(full: pd.DataFrame, solution) -> dict:
    season_means = {}
    for column in solution.DRIFT_COLS:
        means = full.groupby("season", observed=True)[column].mean()
        table = {str(int(year)): float(value) for year, value in means.items()}
        table[str(TARGET_YEAR)] = forecast_linear(means, TARGET_YEAR)
        season_means[column] = table
    teams = sorted(
        set(full.pitcher_team_id.astype(int).unique())
        | set(full.batter_team_id.astype(int).unique())
    )
    return {
        "target_year": TARGET_YEAR,
        "season_means": season_means,
        "pitcher_map": {
            int(value): index
            for index, value in enumerate(sorted(full.pitcher_id.astype(int).unique()))
        },
        "batter_map": {
            int(value): index
            for index, value in enumerate(sorted(full.batter_id.astype(int).unique()))
        },
        "team_map": {int(value): index for index, value in enumerate(teams)},
        "priors": make_priors(full),
    }


def safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(denominator), np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def entity_season_features_training(
    full: pd.DataFrame,
    entity: str,
    n_column: str,
    rate_columns: list[str],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        f"{prefix}_season_n",
        f"{prefix}_season_success_count",
        f"{prefix}_season_success_rate",
        f"{prefix}_season_success_shr25",
        f"{prefix}_season_success_shr75",
        f"{prefix}_prev_season_n",
        f"{prefix}_prev_season_success_rate",
        f"{prefix}_prior_seasons_success_rate",
        f"{prefix}_season_minus_prev",
        f"{prefix}_season_progress",
    ]
    feature_names.extend(
        f"{prefix}_season_{column.removeprefix('asof_')}"
        for column in rate_columns[1:]
    )
    output = {
        name: np.full(len(full), np.nan, dtype=np.float32)
        for name in feature_names
    }
    seasons = sorted(full.season.astype(int).unique())
    for season in seasons:
        mask = full.season.eq(season)
        positions = np.flatnonzero(mask.to_numpy())
        current = full.loc[mask]
        history = full.loc[full.season.lt(season)]
        if len(history):
            history_grouped = history.groupby(entity, observed=True, sort=False)
            history_n = history_grouped.size()
            history_success = history_grouped.control_success.sum()
            prior_rate = history_success / history_n
            previous = history.loc[history.season.eq(season - 1)]
            previous_grouped = previous.groupby(entity, observed=True, sort=False)
            previous_n = previous_grouped.size()
            previous_rate = previous_grouped.control_success.mean()
            last_rows = history_grouped.tail(1).set_index(entity)
            league_history_rate = float(history.control_success.mean())
        else:
            history_n = pd.Series(dtype=float)
            history_success = pd.Series(dtype=float)
            prior_rate = pd.Series(dtype=float)
            previous_n = pd.Series(dtype=float)
            previous_rate = pd.Series(dtype=float)
            last_rows = pd.DataFrame()
            league_history_rate = 0.5

        entity_ids = current[entity]
        start_n = entity_ids.map(history_n).fillna(0).to_numpy(np.float64)
        start_success = entity_ids.map(history_success).fillna(0).to_numpy(np.float64)
        asof_n = current[n_column].to_numpy(np.float64)
        asof_success_count = np.rint(
            np.nan_to_num(current[rate_columns[0]].to_numpy(np.float64) * asof_n, nan=0.0)
        )
        season_n = np.maximum(asof_n - start_n, 0.0)
        season_success = np.maximum(asof_success_count - start_success, 0.0)
        season_rate = safe_rate(season_success, season_n)
        previous_n_values = entity_ids.map(previous_n).fillna(0).to_numpy(np.float64)
        previous_rate_values = entity_ids.map(previous_rate).to_numpy(np.float64)
        prior_rate_values = entity_ids.map(prior_rate).to_numpy(np.float64)
        shrink_prior = np.where(
            np.isfinite(previous_rate_values),
            previous_rate_values,
            np.where(
                np.isfinite(prior_rate_values),
                prior_rate_values,
                league_history_rate,
            ),
        )

        output[f"{prefix}_season_n"][positions] = season_n
        output[f"{prefix}_season_success_count"][positions] = season_success
        output[f"{prefix}_season_success_rate"][positions] = season_rate
        output[f"{prefix}_season_success_shr25"][positions] = (
            season_success + 25.0 * shrink_prior
        ) / (season_n + 25.0)
        output[f"{prefix}_season_success_shr75"][positions] = (
            season_success + 75.0 * shrink_prior
        ) / (season_n + 75.0)
        output[f"{prefix}_prev_season_n"][positions] = previous_n_values
        output[f"{prefix}_prev_season_success_rate"][positions] = previous_rate_values
        output[f"{prefix}_prior_seasons_success_rate"][positions] = prior_rate_values
        output[f"{prefix}_season_minus_prev"][positions] = season_rate - previous_rate_values
        output[f"{prefix}_season_progress"][positions] = season_n / (previous_n_values + 50.0)

        for column in rate_columns[1:]:
            if len(last_rows):
                baseline_series = np.rint(
                    np.nan_to_num(
                        last_rows[column].to_numpy(np.float64)
                        * last_rows[n_column].to_numpy(np.float64),
                        nan=0.0,
                    )
                )
                baseline_lookup = pd.Series(baseline_series, index=last_rows.index)
                baseline = entity_ids.map(baseline_lookup).fillna(0).to_numpy(np.float64)
            else:
                baseline = np.zeros(len(current), dtype=np.float64)
            asof_category = np.rint(
                np.nan_to_num(current[column].to_numpy(np.float64) * asof_n, nan=0.0)
            )
            season_category = np.maximum(asof_category - baseline, 0.0)
            name = f"{prefix}_season_{column.removeprefix('asof_')}"
            output[name][positions] = safe_rate(season_category, season_n)

    return (
        np.column_stack([output[name] for name in feature_names]).astype(np.float32),
        feature_names,
    )


def fit_entity_history(
    full: pd.DataFrame,
    entity: str,
    n_column: str,
    rate_columns: list[str],
) -> dict:
    grouped = full.groupby(entity, observed=True, sort=False)
    n = grouped.size()
    success = grouped.control_success.sum()
    previous = full.loc[full.season.eq(VALID_YEAR)]
    previous_grouped = previous.groupby(entity, observed=True, sort=False)
    last_rows = grouped.tail(1).set_index(entity)
    category_counts = {}
    for column in rate_columns[1:]:
        values = np.rint(
            np.nan_to_num(
                last_rows[column].to_numpy(np.float64)
                * last_rows[n_column].to_numpy(np.float64),
                nan=0.0,
            )
        )
        category_counts[column] = {
            int(key): float(value)
            for key, value in zip(last_rows.index.to_numpy(), values)
        }
    return {
        "n": {int(key): int(value) for key, value in n.items()},
        "success": {int(key): int(value) for key, value in success.items()},
        "prior_rate": {
            int(key): float(value) for key, value in (success / n).items()
        },
        "previous_n": {
            int(key): int(value) for key, value in previous_grouped.size().items()
        },
        "previous_rate": {
            int(key): float(value)
            for key, value in previous_grouped.control_success.mean().items()
        },
        "category_counts": category_counts,
    }


def temporal_target_encodings_training(
    full: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    arrays = {
        name: np.full(len(full), np.nan, dtype=np.float32)
        for name, _, _ in TE_DEFINITIONS
    }
    counts = {
        f"{name}_logn": np.full(len(full), np.nan, dtype=np.float32)
        for name, _, _ in TE_DEFINITIONS
    }
    for season in sorted(full.season.astype(int).unique()):
        mask = full.season.eq(season)
        history = full.loc[full.season.lt(season)]
        if not len(history):
            continue
        current = full.loc[mask]
        positions = np.flatnonzero(mask.to_numpy())
        prior = float(history.control_success.mean())
        for name, keys, alpha in TE_DEFINITIONS:
            stats = history.groupby(keys, observed=True).control_success.agg(["sum", "count"])
            current_index = pd.MultiIndex.from_frame(current[keys])
            numerators = stats["sum"].reindex(current_index).to_numpy(np.float64)
            denominators = stats["count"].reindex(current_index).to_numpy(np.float64)
            arrays[name][positions] = (
                (numerators + alpha * prior) / (denominators + alpha)
            ).astype(np.float32)
            counts[f"{name}_logn"][positions] = np.log1p(denominators).astype(np.float32)
    names: list[str] = []
    pieces: list[np.ndarray] = []
    for name, _, _ in TE_DEFINITIONS:
        names.extend([name, f"{name}_logn"])
        pieces.extend([arrays[name], counts[f"{name}_logn"]])
    return np.column_stack(pieces).astype(np.float32), names


def fit_target_encoding_meta(full: pd.DataFrame) -> dict:
    result = {}
    for name, keys, alpha in TE_DEFINITIONS:
        stats = full.groupby(keys, observed=True).control_success.agg(["sum", "count"])
        sums = {}
        counts = {}
        for key, row in stats.iterrows():
            normalized_key = key if isinstance(key, tuple) else (key,)
            normalized_key = tuple(
                value.item() if hasattr(value, "item") else value
                for value in normalized_key
            )
            sums[normalized_key] = float(row["sum"])
            counts[normalized_key] = int(row["count"])
        result[name] = {
            "alpha": float(alpha),
            "sum": sums,
            "count": counts,
        }
    return result


def temporal_rich_decay_encodings_training(
    full: pd.DataFrame,
    solution,
    half_life: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    """Build time-safe interaction encodings from seasons strictly before Y."""
    names = [
        feature
        for name, _, _ in solution.RICH_TE_SPECS
        for feature in (f"{name}_decay", f"{name}_decay_logn")
    ]
    output = {name: np.full(len(full), np.nan, dtype=np.float32) for name in names}
    target = full.control_success.to_numpy(np.float64)
    for season in sorted(full.season.astype(int).unique()):
        current_mask = full.season.eq(season).to_numpy()
        history_mask = full.season.lt(season).to_numpy()
        if not history_mask.any():
            continue
        current = full.loc[current_mask]
        positions = np.flatnonzero(current_mask)
        history = full.loc[history_mask].copy()
        history["_weight"] = np.power(
            0.5,
            (season - 1 - history.season.to_numpy(np.float64)) / half_life,
        )
        history["_weighted_target"] = target[history_mask] * history["_weight"]
        prior = float(history._weighted_target.sum() / history._weight.sum())
        for name, keys, alpha in solution.RICH_TE_SPECS:
            stats = history.groupby(keys, observed=True, sort=False).agg(
                weighted_sum=("_weighted_target", "sum"),
                weighted_count=("_weight", "sum"),
            )
            current_index = pd.MultiIndex.from_frame(current[keys])
            looked_up = stats.reindex(current_index)
            numerator = looked_up.weighted_sum.to_numpy(np.float64)
            denominator = looked_up.weighted_count.to_numpy(np.float64)
            output[f"{name}_decay"][positions] = (
                (numerator + alpha * prior) / (denominator + alpha)
            ).astype(np.float32)
            output[f"{name}_decay_logn"][positions] = np.log1p(denominator).astype(
                np.float32
            )
    return np.column_stack([output[name] for name in names]).astype(np.float32), names


def fit_rich_decay_meta(
    full: pd.DataFrame,
    solution,
    half_life: float = 2.0,
) -> tuple[dict, float]:
    """Freeze the 2025 row-wise lookup tables for rich decayed encodings."""
    work = full.copy()
    work["_weight"] = np.power(
        0.5,
        (TARGET_YEAR - 1 - work.season.to_numpy(np.float64)) / half_life,
    )
    work["_weighted_target"] = (
        work.control_success.to_numpy(np.float64) * work["_weight"].to_numpy(np.float64)
    )
    prior = float(work._weighted_target.sum() / work._weight.sum())
    result = {}
    for name, keys, _ in solution.RICH_TE_SPECS:
        stats = work.groupby(keys, observed=True, sort=False).agg(
            weighted_sum=("_weighted_target", "sum"),
            weighted_count=("_weight", "sum"),
        )
        weighted_sums = {}
        weighted_counts = {}
        for key, row in stats.iterrows():
            normalized_key = key if isinstance(key, tuple) else (key,)
            normalized_key = tuple(
                value.item() if hasattr(value, "item") else value
                for value in normalized_key
            )
            weighted_sums[normalized_key] = float(row["weighted_sum"])
            weighted_counts[normalized_key] = float(row["weighted_count"])
        result[name] = {
            "weighted_sum": weighted_sums,
            "weighted_count": weighted_counts,
        }
    return result, prior


def main_pitch_fingerprints(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build historical pitch-mix fingerprints from official pre-pitch rows.

    Only the first and last official ``asof_*`` snapshots within a
    pitcher-season are subtracted.  In particular, this does not inspect the
    next row to reconstruct the current pitch type.
    """
    frame = train[
        [
            "pitcher_id",
            "pitcher_hand",
            "season",
            "asof_pitcher_pitchmix_n",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].copy()
    keys = ["pitcher_id", "season"]
    order_columns = [*keys, "asof_pitcher_pitchmix_n"]
    if frame.duplicated(order_columns).any():
        raise RuntimeError("duplicate pitcher-season pitch-mix snapshot")
    ordered = frame.sort_values(order_columns, kind="mergesort")
    first = ordered.groupby(keys, observed=True, sort=False).head(1).set_index(keys)
    last = ordered.groupby(keys, observed=True, sort=False).tail(1).set_index(keys)
    start_n = first.asof_pitcher_pitchmix_n.reindex(last.index).to_numpy(np.float64)
    end_n = last.asof_pitcher_pitchmix_n.to_numpy(np.float64)
    snapshot_delta = end_n - start_n
    if (snapshot_delta < 0).any():
        raise RuntimeError("pitch-mix snapshot count decreased within a season")
    aggregates = last.reset_index()[["pitcher_id", "season"]].copy()
    aggregates["n"] = snapshot_delta
    for group in PITCH_GROUPS[:3]:
        column = f"asof_pitcher_{group}_rate"
        start_count = np.rint(
            np.nan_to_num(
                first[column].reindex(last.index).to_numpy(np.float64) * start_n,
                nan=0.0,
            )
        )
        end_count = np.rint(
            np.nan_to_num(last[column].to_numpy(np.float64) * end_n, nan=0.0)
        )
        count_delta = end_count - start_count
        if (count_delta < 0).any():
            raise RuntimeError(f"{group} pitch count decreased within a season")
        aggregates[group] = count_delta
    if not np.array_equal(
        aggregates[PITCH_GROUPS[:3]].sum(axis=1).to_numpy(np.float64),
        aggregates["n"].to_numpy(np.float64),
    ):
        raise RuntimeError("pitch-mix endpoint counts do not sum to snapshot delta")
    aggregates["pitcher_hand"] = last.pitcher_hand.map({1: "Left", 2: "Right"}).to_numpy()
    hand_counts = (
        aggregates.groupby(["pitcher_id", "pitcher_hand"], observed=True, sort=False)
        .n.sum()
        .reset_index()
        .sort_values(
            ["pitcher_id", "n", "pitcher_hand"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    hands = hand_counts.drop_duplicates("pitcher_id").set_index("pitcher_id").pitcher_hand
    return aggregates, hands


def trackman_pitch_fingerprints(trackman: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    valid = trackman.loc[trackman.pitch_type_group.isin(PITCH_GROUPS[:3])]
    aggregates = pd.crosstab(
        [valid.pitcher_trackman_id, valid.season], valid.pitch_type_group
    ).reset_index()
    for group in PITCH_GROUPS[:3]:
        if group not in aggregates:
            aggregates[group] = 0
    aggregates["n"] = aggregates[PITCH_GROUPS[:3]].sum(axis=1)
    hand_counts = (
        valid.groupby(
            ["pitcher_trackman_id", "season", "pitcher_hand"],
            observed=True,
            sort=False,
        )
        .size()
        .rename("hand_n")
        .reset_index()
        .sort_values(
            ["pitcher_trackman_id", "season", "hand_n", "pitcher_hand"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["pitcher_trackman_id", "season"])
    )
    aggregates = aggregates.merge(
        hand_counts[["pitcher_trackman_id", "season", "pitcher_hand"]],
        on=["pitcher_trackman_id", "season"],
        how="left",
        validate="one_to_one",
    )
    overall_hand_counts = (
        hand_counts.groupby(
            ["pitcher_trackman_id", "pitcher_hand"], observed=True, sort=False
        )
        .hand_n.sum()
        .reset_index()
        .sort_values(
            ["pitcher_trackman_id", "hand_n", "pitcher_hand"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    hands = (
        overall_hand_counts.drop_duplicates("pitcher_trackman_id")
        .set_index("pitcher_trackman_id")
        .pitcher_hand
    )
    return aggregates, hands


def resolve_pitchers(
    main_agg: pd.DataFrame,
    main_hands: pd.Series,
    track_agg: pd.DataFrame,
    track_hands: pd.Series,
    cutoff: int,
    assignment: str = "local",
    assignment_slack: float = 0.35,
    include_collision_features: bool = False,
) -> pd.DataFrame:
    if "pitcher_hand" in main_agg.columns:
        main_hand_counts = (
            main_agg.groupby(
                ["pitcher_id", "pitcher_hand"], observed=True, sort=False
            )
            .n.sum()
            .reset_index()
            .sort_values(
                ["pitcher_id", "n", "pitcher_hand"],
                ascending=[True, False, True],
                kind="mergesort",
            )
        )
        main_hands = (
            main_hand_counts.drop_duplicates("pitcher_id")
            .set_index("pitcher_id")
            .pitcher_hand
        )
    if "pitcher_hand" in track_agg.columns:
        track_hand_counts = (
            track_agg.groupby(
                ["pitcher_trackman_id", "pitcher_hand"],
                observed=True,
                sort=False,
            )
            .n.sum()
            .reset_index()
            .sort_values(
                ["pitcher_trackman_id", "n", "pitcher_hand"],
                ascending=[True, False, True],
                kind="mergesort",
            )
        )
        track_hands = (
            track_hand_counts.drop_duplicates("pitcher_trackman_id")
            .set_index("pitcher_trackman_id")
            .pitcher_hand
        )
    seasons = np.arange(2019, cutoff + 1)
    main_ids = sorted(main_agg.pitcher_id.unique())
    track_ids = sorted(track_agg.pitcher_trackman_id.unique())
    main_index = {value: index for index, value in enumerate(main_ids)}
    track_index = {value: index for index, value in enumerate(track_ids)}
    count_main = np.zeros((len(main_ids), len(seasons)))
    count_track = np.zeros((len(track_ids), len(seasons)))
    rate_main = np.zeros((len(main_ids), len(seasons), 3))
    rate_track = np.zeros((len(track_ids), len(seasons), 3))
    for row in main_agg.itertuples():
        i = main_index[row.pitcher_id]
        j = int(row.season) - 2019
        values = np.array([row.fastball, row.breaking, row.offspeed])
        count_main[i, j] = row.n
        rate_main[i, j] = values / max(row.n, 1)
    for row in track_agg.itertuples():
        i = track_index[row.pitcher_trackman_id]
        j = int(row.season) - 2019
        values = np.array([row.fastball, row.breaking, row.offspeed])
        count_track[i, j] = row.n
        rate_track[i, j] = values / max(row.n, 1)

    scores = np.full((len(main_ids), len(track_ids)), np.inf, dtype=np.float32)
    details = {}
    for i, main_id in enumerate(main_ids):
        if count_main[i].sum() < 50:
            continue
        for j, track_id in enumerate(track_ids):
            if track_hands.get(track_id) != main_hands.get(main_id) or count_track[j].sum() < 30:
                continue
            overlap = (count_main[i] >= 20) & (count_track[j] >= 20)
            if not overlap.any():
                continue
            weight = np.minimum(
                np.minimum(count_main[i, overlap], count_track[j, overlap]), 300
            )
            mix_rms = np.sqrt(
                np.average(
                    np.mean((rate_main[i, overlap] - rate_track[j, overlap]) ** 2, axis=1),
                    weights=weight,
                )
            )
            log_ratio = np.log(
                (count_track[j, overlap] + 20) / (count_main[i, overlap] + 20)
            )
            count_shape = np.std(log_ratio) if overlap.sum() > 1 else 0.35
            active_main = count_main[i] >= 20
            active_track = count_track[j] >= 20
            jaccard = (active_main & active_track).sum() / max(
                (active_main | active_track).sum(), 1
            )
            total_ratio = abs(
                np.log((count_track[j].sum() + 100) / (count_main[i].sum() + 100))
            )
            score = (
                mix_rms / 0.08
                + count_shape / 0.55
                + (1 - jaccard) * 1.2
                + max(0, total_ratio - 1.0) * 0.15
            )
            scores[i, j] = score
            details[(i, j)] = (mix_rms, count_shape, jaccard, overlap.sum())

    local_best_j = np.argmin(scores, axis=1)
    local_best_score = scores[np.arange(len(main_ids)), local_best_j]
    best_i = np.argmin(scores, axis=0)
    if assignment == "local":
        selected_j = local_best_j
    elif assignment == "global":
        # Assign every Trackman identity to at most one main-data pitcher.  A
        # row-specific dummy option keeps an ambiguous collision unmatched
        # unless a viable alternative is close to that pitcher's local best.
        n_main, n_track = scores.shape
        finite = np.isfinite(local_best_score)
        large_cost = np.float64(1e6)
        costs = np.full((n_main, n_track + n_main), large_cost, dtype=np.float64)
        costs[:, :n_track] = np.where(np.isfinite(scores), scores, large_cost)
        for i in range(n_main):
            if finite[i]:
                costs[i, n_track + i] = float(local_best_score[i] + assignment_slack)
            else:
                costs[i, n_track + i] = 0.0
        row_indices, column_indices = linear_sum_assignment(costs)
        selected_j = np.full(n_main, -1, dtype=np.int64)
        for i, column in zip(row_indices, column_indices):
            if column < n_track and costs[i, column] < large_cost:
                selected_j[i] = int(column)
    else:
        raise ValueError(f"unknown assignment mode: {assignment}")

    rows = []
    for i, j in enumerate(selected_j):
        if j < 0 or not np.isfinite(scores[i, j]):
            continue
        mix_rms, count_shape, jaccard, overlap = details[(i, j)]
        alternatives = scores[i].copy()
        alternatives[j] = np.inf
        alternative_score = float(np.min(alternatives))
        selected_score = float(scores[i, j])
        rows.append(
            {
                "pitcher_id": int(main_ids[i]),
                "pitcher_trackman_id": int(track_ids[j]),
                "tm_map_score": selected_score,
                "tm_map_margin": float(alternative_score - selected_score),
                "tm_map_reciprocal": float(best_i[j] == i),
                "tm_map_mix_rms": float(mix_rms),
                "tm_map_count_shape": float(count_shape),
                "tm_map_jaccard": float(jaccard),
                "tm_map_overlap_years": float(overlap),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) and include_collision_features:
        multiplicity = result.pitcher_trackman_id.map(
            result.pitcher_trackman_id.value_counts()
        ).astype(np.float32)
        result["tm_map_multiplicity"] = multiplicity
        result["tm_map_is_collision"] = multiplicity.gt(1).astype(np.float32)
    return result


def build_trackman_profiles(trackman: pd.DataFrame, cutoffs: list[int]) -> dict[int, pd.DataFrame]:
    work = trackman[
        ["pitcher_trackman_id", "season", "pitch_type_group"] + MEASURES
    ].copy()
    for measure in MEASURES:
        work[f"{measure}_sq"] = work[measure] ** 2
    aggregations = {"pitch_count": ("season", "size")}
    for measure in MEASURES:
        aggregations[f"{measure}_count"] = (measure, "count")
        aggregations[f"{measure}_sum"] = (measure, "sum")
        aggregations[f"{measure}_sumsq"] = (f"{measure}_sq", "sum")
    yearly = (
        work.groupby(
            ["pitcher_trackman_id", "season", "pitch_type_group"], observed=True
        )
        .agg(**aggregations)
        .reset_index()
    )
    profiles = {}
    for cutoff in cutoffs:
        selected = yearly.loc[yearly.season.le(cutoff)]
        grouped = selected.groupby(
            ["pitcher_trackman_id", "pitch_type_group"], observed=True
        ).sum(numeric_only=True)
        ids = sorted(selected.pitcher_trackman_id.unique())
        profile = pd.DataFrame(index=pd.Index(ids, name="pitcher_trackman_id"))
        total_count = grouped.pitch_count.groupby(level=0).sum()
        profile["tm_log_n"] = np.log1p(total_count)
        for group in PITCH_GROUPS:
            if group not in grouped.index.get_level_values(1):
                continue
            group_rows = grouped.xs(group, level=1)
            profile[f"tm_{group}_share"] = group_rows.pitch_count / total_count
            for measure in MEASURES:
                count = group_rows[f"{measure}_count"]
                mean = group_rows[f"{measure}_sum"] / count.replace(0, np.nan)
                variance = (
                    group_rows[f"{measure}_sumsq"] / count.replace(0, np.nan)
                    - mean**2
                ).clip(lower=0)
                profile[f"tm_{group}_{measure}_mean"] = mean
                profile[f"tm_{group}_{measure}_std"] = np.sqrt(variance)
        profiles[cutoff] = profile.astype(np.float32)
    all_columns = sorted(set().union(*(profile.columns for profile in profiles.values())))
    return {
        cutoff: profile.reindex(columns=all_columns)
        for cutoff, profile in profiles.items()
    }


def build_trackman_temporal_profiles(
    trackman: pd.DataFrame,
    cutoffs: list[int],
) -> dict[int, pd.DataFrame]:
    """Build TMT-1 pitcher trajectories using only data through each cutoff.

    A pitcher's latest observed season is compared with all strictly earlier
    seasons.  Measurement means are first standardised within TrackMan
    season-by-pitch-group, which removes year-level measurement drift.  The
    earlier-career reference is observation-count weighted and excludes the
    latest season entirely.  No target or official evaluation row is read.
    """
    required = {
        "pitcher_trackman_id",
        "season",
        "pitch_type_group",
        *MEASURES,
    }
    missing = sorted(required - set(trackman.columns))
    if missing:
        raise ValueError(f"TrackMan is missing TMT-1 columns: {missing}")
    if not cutoffs:
        raise ValueError("TMT-1 requires at least one cutoff")

    work = trackman.loc[
        trackman.pitch_type_group.isin(PITCH_GROUPS),
        ["pitcher_trackman_id", "season", "pitch_type_group", *MEASURES],
    ].copy()
    aggregations: dict[str, tuple[str, str]] = {"pitch_n": ("season", "size")}
    for measure in MEASURES:
        square = f"__tmt_{measure}_sq"
        work[square] = work[measure] ** 2
        aggregations[f"{measure}_n"] = (measure, "count")
        aggregations[f"{measure}_sum"] = (measure, "sum")
        aggregations[f"{measure}_sumsq"] = (square, "sum")
    yearly = (
        work.groupby(
            ["pitcher_trackman_id", "season", "pitch_type_group"],
            observed=True,
            sort=False,
        )
        .agg(**aggregations)
        .reset_index()
    )
    del work

    global_rows = yearly.groupby(
        ["season", "pitch_type_group"], observed=True, sort=False
    ).sum(numeric_only=True)
    global_columns: list[str] = []
    for measure in MEASURES:
        count = global_rows[f"{measure}_n"].replace(0, np.nan)
        mean = global_rows[f"{measure}_sum"] / count
        variance = (global_rows[f"{measure}_sumsq"] / count - mean**2).clip(
            lower=0
        )
        mean_column = f"{measure}_global_mean"
        std_column = f"{measure}_global_std"
        global_rows[mean_column] = mean
        global_rows[std_column] = np.sqrt(variance).replace(0, np.nan)
        global_columns.extend([mean_column, std_column])
    yearly = yearly.merge(
        global_rows[global_columns].reset_index(),
        on=["season", "pitch_type_group"],
        how="left",
        validate="many_to_one",
    )
    for measure in MEASURES:
        count = yearly[f"{measure}_n"].replace(0, np.nan)
        mean = yearly[f"{measure}_sum"] / count
        yearly[f"{measure}_zmean"] = (
            mean - yearly[f"{measure}_global_mean"]
        ) / yearly[f"{measure}_global_std"]

    profiles: dict[int, pd.DataFrame] = {}
    for cutoff in sorted(set(map(int, cutoffs))):
        history = yearly.loc[yearly.season.le(cutoff)].copy()
        if history.empty:
            profiles[cutoff] = pd.DataFrame(
                columns=TMT_PROFILE_COLUMNS,
                index=pd.Index([], name="pitcher_trackman_id"),
                dtype=np.float32,
            )
            continue
        latest_season = history.groupby(
            "pitcher_trackman_id", observed=True, sort=False
        ).season.max()
        history = history.merge(
            latest_season.rename("latest_season"),
            left_on="pitcher_trackman_id",
            right_index=True,
            validate="many_to_one",
        )
        last = history.loc[history.season.eq(history.latest_season)].copy()
        prior = history.loc[history.season.lt(history.latest_season)].copy()
        ids = pd.Index(
            sorted(history.pitcher_trackman_id.unique()),
            name="pitcher_trackman_id",
        )
        profile = pd.DataFrame(index=ids)
        last_total = last.groupby(
            "pitcher_trackman_id", observed=True, sort=False
        ).pitch_n.sum()
        prior_total = prior.groupby(
            "pitcher_trackman_id", observed=True, sort=False
        ).pitch_n.sum()
        profile["tmt_last_log_n"] = np.log1p(last_total.reindex(ids).fillna(0.0))
        profile["tmt_prior_log_n"] = np.log1p(prior_total.reindex(ids).fillna(0.0))
        profile["tmt_profile_age"] = cutoff - latest_season.reindex(ids)
        profile["tmt_two_season_available"] = (
            prior_total.reindex(ids).fillna(0.0).gt(0).astype(np.float32)
        )

        last_group_counts = last.pivot_table(
            index="pitcher_trackman_id",
            columns="pitch_type_group",
            values="pitch_n",
            aggfunc="sum",
            fill_value=0.0,
        )
        prior_group_counts = prior.pivot_table(
            index="pitcher_trackman_id",
            columns="pitch_type_group",
            values="pitch_n",
            aggfunc="sum",
            fill_value=0.0,
        )
        for group in TMT_GROUPS:
            last_count = (
                last_group_counts[group]
                if group in last_group_counts
                else pd.Series(0.0, index=last_group_counts.index)
            )
            prior_count = (
                prior_group_counts[group]
                if group in prior_group_counts
                else pd.Series(0.0, index=prior_group_counts.index)
            )
            last_share = last_count.reindex(ids).fillna(0.0) / last_total.reindex(
                ids
            ).replace(0, np.nan)
            prior_share = prior_count.reindex(ids).fillna(0.0) / prior_total.reindex(
                ids
            ).replace(0, np.nan)
            profile[f"tmt_{group}_share_delta"] = last_share - prior_share

            last_group = last.loc[last.pitch_type_group.eq(group)].set_index(
                "pitcher_trackman_id"
            )
            prior_group = prior.loc[prior.pitch_type_group.eq(group)].copy()
            for measure in MEASURES:
                weighted_column = f"__tmt_{measure}_weighted_z"
                prior_group[weighted_column] = (
                    prior_group[f"{measure}_zmean"]
                    * prior_group[f"{measure}_n"]
                )
                grouped_prior = prior_group.groupby(
                    "pitcher_trackman_id", observed=True, sort=False
                )
                prior_weight = grouped_prior[f"{measure}_n"].sum().replace(0, np.nan)
                prior_z = grouped_prior[weighted_column].sum() / prior_weight
                last_z = (
                    last_group[f"{measure}_zmean"]
                    if f"{measure}_zmean" in last_group
                    else pd.Series(dtype=np.float64)
                )
                profile[f"tmt_{group}_{measure}_z_delta"] = (
                    last_z.reindex(ids) - prior_z.reindex(ids)
                )
                del prior_group[weighted_column]
        profile = profile.reindex(columns=TMT_PROFILE_COLUMNS)
        if profile.columns.tolist() != TMT_PROFILE_COLUMNS:
            raise RuntimeError("TMT-1 profile column order changed")
        profiles[cutoff] = profile.astype(np.float32)
    return profiles


def attach_trackman_temporal_training(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, list[str]]:
    """Attach a frozen TMT-1 table built through ``row season - 1``."""
    if not profiles:
        raise ValueError("TMT-1 training attachment requires profiles")
    map_frame = mapping.set_index("pitcher_id")
    if not map_frame.index.is_unique:
        raise ValueError("TMT-1 mapping has duplicate pitcher_id values")
    mapped_id = frame.pitcher_id.map(map_frame.pitcher_trackman_id)
    arrays = {
        column: np.full(len(frame), np.nan, dtype=np.float32)
        for column in TMT_PROFILE_COLUMNS
    }
    for season in sorted(frame.season.astype(int).unique()):
        cutoff = int(season) - 1
        if cutoff not in profiles:
            continue
        mask = frame.season.eq(season).to_numpy()
        ids = mapped_id.loc[mask]
        profile = profiles[cutoff]
        for column in TMT_PROFILE_COLUMNS:
            arrays[column][mask] = ids.map(profile[column]).to_numpy(np.float32)
    return (
        np.column_stack([arrays[column] for column in TMT_PROFILE_COLUMNS]).astype(
            np.float32
        ),
        list(TMT_PROFILE_COLUMNS),
    )


def fit_trackman_temporal_meta(
    target_profile: pd.DataFrame,
    cutoff: int,
) -> dict:
    """Freeze the TMT-1 lookup used for one-row-at-a-time inference."""
    if target_profile.columns.tolist() != TMT_PROFILE_COLUMNS:
        raise ValueError("TMT-1 target profile has an unexpected layout")
    return {
        "schema_version": 1,
        "cutoff": int(cutoff),
        "groups": list(TMT_GROUPS),
        "measures": list(MEASURES),
        "definition": "latest observed season z minus strictly earlier career z",
        "profile_columns": list(TMT_PROFILE_COLUMNS),
        "profile_values": {
            column: {
                int(key): float(value)
                for key, value in target_profile[column].dropna().items()
            }
            for column in TMT_PROFILE_COLUMNS
        },
    }


def _cst_snapshot_counts(
    frame: pd.DataFrame,
    n_column: str,
    rate_columns: tuple[str, ...],
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Return the last legal cumulative snapshot for each pitcher."""
    if frame.empty:
        empty = pd.Series(dtype=np.float64)
        return empty, {column: empty.copy() for column in rate_columns}
    ordered = frame.sort_values(
        ["pitcher_id", n_column], kind="mergesort"
    )
    if ordered.duplicated(["pitcher_id", n_column]).any():
        raise ValueError(f"CST-1 duplicate cumulative state for {n_column}")
    last = ordered.groupby(
        "pitcher_id", observed=True, sort=False
    ).tail(1).set_index("pitcher_id")
    denominator = last[n_column].astype(np.float64)
    counts: dict[str, pd.Series] = {}
    for column in rate_columns:
        rate = last[column].astype(np.float64)
        values = np.where(
            np.isfinite(rate),
            np.rint(rate.to_numpy(np.float64) * denominator.to_numpy(np.float64)),
            np.nan,
        )
        counts[column] = pd.Series(values, index=last.index, dtype=np.float64)
    return denominator, counts


def _cst_previous_season_profile(
    frame: pd.DataFrame,
    n_column: str,
    rate_columns: tuple[str, ...],
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Recover one season's rates from its first and last pre-pitch states."""
    if frame.empty:
        empty = pd.Series(dtype=np.float64)
        return empty, {column: empty.copy() for column in rate_columns}
    ordered = frame.sort_values(
        ["pitcher_id", n_column], kind="mergesort"
    )
    if ordered.duplicated(["pitcher_id", n_column]).any():
        raise ValueError(f"CST-1 duplicate season state for {n_column}")
    grouped = ordered.groupby("pitcher_id", observed=True, sort=False)
    first = grouped.head(1).set_index("pitcher_id")
    last = grouped.tail(1).set_index("pitcher_id")
    first = first.reindex(last.index)
    season_n = (
        last[n_column].astype(np.float64)
        - first[n_column].astype(np.float64)
    )
    if season_n.lt(0).any():
        raise ValueError(f"CST-1 {n_column} decreased across a season")
    rates: dict[str, pd.Series] = {}
    for column in rate_columns:
        start_rate = first[column].astype(np.float64)
        end_rate = last[column].astype(np.float64)
        start_count = np.where(
            np.isfinite(start_rate),
            np.rint(
                start_rate.to_numpy(np.float64)
                * first[n_column].to_numpy(np.float64)
            ),
            np.nan,
        )
        end_count = np.where(
            np.isfinite(end_rate),
            np.rint(
                end_rate.to_numpy(np.float64)
                * last[n_column].to_numpy(np.float64)
            ),
            np.nan,
        )
        count_delta = end_count - start_count
        finite = np.isfinite(count_delta)
        if np.any(count_delta[finite] < 0):
            raise ValueError(f"CST-1 {column} count decreased across a season")
        values = np.full(len(last), np.nan, dtype=np.float64)
        season_n_values = season_n.to_numpy(np.float64)
        usable = finite & (season_n_values > 0)
        values[usable] = count_delta[usable] / season_n_values[usable]
        rates[column] = pd.Series(values, index=last.index, dtype=np.float64)
    return season_n, rates


def _cst_freeze(series: pd.Series) -> dict[int, float]:
    return {
        int(key): float(value)
        for key, value in series.dropna().items()
    }


def fit_cst_meta(history: pd.DataFrame, target_year: int) -> dict:
    """Freeze CST-1 state using only rows strictly before ``target_year``.

    The official cumulative ``asof_*`` snapshots are sufficient to recover
    both the previous-season profile and the start counter for the target
    season.  The target column is deliberately not inspected.
    """
    required = {
        "pitcher_id",
        "season",
        "asof_pitcher_n",
        "asof_pitcher_pitchmix_n",
        *CST_COMMAND_RATE_COLUMNS,
        *CST_MIX_RATE_COLUMNS,
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"CST-1 history is missing columns: {missing}")
    target_year = int(target_year)
    work = history.loc[
        history.season.astype(int).lt(target_year),
        sorted(required),
    ].copy()
    previous = work.loc[work.season.astype(int).eq(target_year - 1)].copy()

    command_start_n, command_start_counts = _cst_snapshot_counts(
        work, "asof_pitcher_n", CST_COMMAND_RATE_COLUMNS
    )
    mix_start_n, mix_start_counts = _cst_snapshot_counts(
        work, "asof_pitcher_pitchmix_n", CST_MIX_RATE_COLUMNS
    )
    previous_command_n, previous_command_rates = _cst_previous_season_profile(
        previous, "asof_pitcher_n", CST_COMMAND_RATE_COLUMNS
    )
    previous_mix_n, previous_mix_rates = _cst_previous_season_profile(
        previous, "asof_pitcher_pitchmix_n", CST_MIX_RATE_COLUMNS
    )
    return {
        "schema_version": 1,
        "target_year": target_year,
        "shrinkage_count": float(CST_SHRINKAGE_N),
        "command_rate_columns": list(CST_COMMAND_RATE_COLUMNS),
        "mix_rate_columns": list(CST_MIX_RATE_COLUMNS),
        "profile_columns": list(CST_PROFILE_COLUMNS),
        "current_start": {
            "command_n": _cst_freeze(command_start_n),
            "command_counts": {
                column: _cst_freeze(command_start_counts[column])
                for column in CST_COMMAND_RATE_COLUMNS
            },
            "mix_n": _cst_freeze(mix_start_n),
            "mix_counts": {
                column: _cst_freeze(mix_start_counts[column])
                for column in CST_MIX_RATE_COLUMNS
            },
        },
        "previous": {
            "command_n": _cst_freeze(previous_command_n.where(previous_command_n.gt(0))),
            "command_rates": {
                column: _cst_freeze(previous_command_rates[column])
                for column in CST_COMMAND_RATE_COLUMNS
            },
            "mix_n": _cst_freeze(previous_mix_n.where(previous_mix_n.gt(0))),
            "mix_rates": {
                column: _cst_freeze(previous_mix_rates[column])
                for column in CST_MIX_RATE_COLUMNS
            },
        },
    }


def build_cst_training_features(
    frame: pd.DataFrame,
    solution,
) -> tuple[np.ndarray, list[str]]:
    """Build season-walk-forward CST-1 features with inference parity."""
    matrix = np.full(
        (len(frame), len(CST_PROFILE_COLUMNS)), np.nan, dtype=np.float32
    )
    observed_names: list[str] | None = None
    seasons = sorted(frame.season.astype(int).unique())
    for season in seasons:
        mask = frame.season.astype(int).eq(int(season)).to_numpy()
        positions = np.flatnonzero(mask)
        frozen = fit_cst_meta(
            frame.loc[frame.season.astype(int).lt(int(season))], int(season)
        )
        current_matrix, names = solution.build_cst_features(
            frame.loc[mask], {"command_state_transition": frozen}
        )
        if observed_names is None:
            observed_names = names
        elif names != observed_names:
            raise RuntimeError("CST-1 training feature order changed across seasons")
        matrix[positions] = current_matrix
    names = observed_names or list(CST_PROFILE_COLUMNS)
    if names != CST_PROFILE_COLUMNS:
        raise RuntimeError("CST-1 training feature layout is invalid")
    return matrix, names


def attach_trackman_training(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, list[str]]:
    map_frame = mapping.set_index("pitcher_id")
    mapped_id = frame.pitcher_id.map(map_frame.pitcher_trackman_id)
    mapping_columns = [
        column for column in map_frame.columns if column != "pitcher_trackman_id"
    ]
    pieces = [
        frame.pitcher_id.map(map_frame[column]).to_numpy(np.float32)
        for column in mapping_columns
    ]
    names = list(mapping_columns)
    pieces.append(mapped_id.notna().to_numpy(np.float32))
    names.append("tm_map_available")
    profile_columns = next(iter(profiles.values())).columns.tolist()
    profile_arrays = {
        column: np.full(len(frame), np.nan, dtype=np.float32)
        for column in profile_columns
    }
    for season in sorted(frame.season.astype(int).unique()):
        cutoff = int(season) - 1
        if cutoff not in profiles:
            continue
        mask = frame.season.eq(season).to_numpy()
        profile = profiles[cutoff]
        ids = mapped_id.loc[mask]
        for column in profile_columns:
            profile_arrays[column][mask] = ids.map(profile[column]).to_numpy(np.float32)
    for column in profile_columns:
        pieces.append(profile_arrays[column])
        names.append(column)
    return np.column_stack(pieces).astype(np.float32), names


def fit_trackman_meta(
    mapping: pd.DataFrame,
    target_profile: pd.DataFrame,
    profile_columns: list[str],
) -> dict:
    map_frame = mapping.set_index("pitcher_id")
    mapping_columns = [
        column for column in map_frame.columns if column != "pitcher_trackman_id"
    ]
    return {
        "pitcher_to_trackman": {
            int(key): int(value)
            for key, value in map_frame.pitcher_trackman_id.items()
        },
        "mapping_columns": mapping_columns,
        "mapping_values": {
            column: {
                int(key): float(value) for key, value in map_frame[column].items()
            }
            for column in mapping_columns
        },
        "profile_columns": profile_columns,
        "profile_values": {
            column: {
                int(key): float(value)
                for key, value in target_profile[column].dropna().items()
            }
            for column in profile_columns
        },
    }


def build_batter_trackman_profiles(
    trackman: pd.DataFrame, cutoffs: list[int]
) -> dict[int, pd.DataFrame]:
    """Physical profile of the pitches each batter faced.

    Mirrors build_trackman_profiles on the batter side.  The pitcher path is
    left untouched: it is the code that produced the measured 1099.170, and a
    shared refactor would put that at risk for no benefit.
    """
    work = trackman[
        ["batter_trackman_id", "season", "pitch_type_group"] + MEASURES
    ].copy()
    for measure in MEASURES:
        work[f"{measure}_sq"] = work[measure] ** 2
    aggregations = {"pitch_count": ("season", "size")}
    for measure in MEASURES:
        aggregations[f"{measure}_count"] = (measure, "count")
        aggregations[f"{measure}_sum"] = (measure, "sum")
        aggregations[f"{measure}_sumsq"] = (f"{measure}_sq", "sum")
    yearly = (
        work.groupby(
            ["batter_trackman_id", "season", "pitch_type_group"], observed=True
        )
        .agg(**aggregations)
        .reset_index()
    )
    profiles = {}
    for cutoff in cutoffs:
        selected = yearly.loc[yearly.season.le(cutoff)]
        grouped = selected.groupby(
            ["batter_trackman_id", "pitch_type_group"], observed=True
        ).sum(numeric_only=True)
        ids = sorted(selected.batter_trackman_id.unique())
        profile = pd.DataFrame(index=pd.Index(ids, name="batter_trackman_id"))
        total_count = grouped.pitch_count.groupby(level=0).sum()
        profile["tmb_log_n"] = np.log1p(total_count)
        for group in PITCH_GROUPS:
            if group not in grouped.index.get_level_values(1):
                continue
            group_rows = grouped.xs(group, level=1)
            profile[f"tmb_{group}_share"] = group_rows.pitch_count / total_count
            for measure in MEASURES:
                count = group_rows[f"{measure}_count"]
                mean = group_rows[f"{measure}_sum"] / count.replace(0, np.nan)
                variance = (
                    group_rows[f"{measure}_sumsq"] / count.replace(0, np.nan)
                    - mean**2
                ).clip(lower=0)
                profile[f"tmb_{group}_{measure}_mean"] = mean
                profile[f"tmb_{group}_{measure}_std"] = np.sqrt(variance)
        profiles[cutoff] = profile.astype(np.float32)
    all_columns = sorted(set().union(*(profile.columns for profile in profiles.values())))
    return {
        cutoff: profile.reindex(columns=all_columns)
        for cutoff, profile in profiles.items()
    }


def attach_batter_trackman_training(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, list[str]]:
    """Season Y rows read the profile built from TrackMan through Y-1."""
    map_frame = mapping.set_index("batter_id")
    mapped_id = frame.batter_id.map(map_frame.batter_trackman_id)
    pieces = [mapped_id.notna().to_numpy(np.float32)]
    names = ["tmb_map_available"]
    profile_columns = next(iter(profiles.values())).columns.tolist()
    profile_arrays = {
        column: np.full(len(frame), np.nan, dtype=np.float32)
        for column in profile_columns
    }
    for season in sorted(frame.season.astype(int).unique()):
        cutoff = int(season) - 1
        if cutoff not in profiles:
            continue
        mask = frame.season.eq(season).to_numpy()
        profile = profiles[cutoff]
        ids = mapped_id.loc[mask]
        for column in profile_columns:
            profile_arrays[column][mask] = ids.map(profile[column]).to_numpy(np.float32)
    for column in profile_columns:
        pieces.append(profile_arrays[column])
        names.append(column)
    return np.column_stack(pieces).astype(np.float32), names


def fit_batter_trackman_meta(
    mapping: pd.DataFrame,
    target_profile: pd.DataFrame,
    profile_columns: list[str],
) -> dict:
    map_frame = mapping.set_index("batter_id")
    return {
        "batter_to_trackman": {
            int(key): int(value)
            for key, value in map_frame.batter_trackman_id.items()
        },
        "profile_columns": profile_columns,
        "profile_values": {
            column: {
                int(key): float(value)
                for key, value in target_profile[column].dropna().items()
            }
            for column in profile_columns
        },
    }


def catboost_frame(
    matrix: np.ndarray, names: list[str], cat_indices: list[int]
) -> pd.DataFrame:
    result = pd.DataFrame(matrix, columns=names)
    for index in cat_indices:
        column = names[index]
        result[column] = result[column].fillna(-1).astype(np.int32)
    return result


def logloss(y: np.ndarray, prediction: np.ndarray) -> float:
    prediction = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(prediction) + (1 - y) * np.log1p(-prediction)))


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((prediction - y) ** 2))


def auc(y: np.ndarray, prediction: np.ndarray) -> float:
    order = np.argsort(prediction, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    positive = y == 1
    n_positive = positive.sum()
    n_negative = len(y) - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def solve_offset(prediction: np.ndarray, target_mean: float) -> float:
    logits = np.log(
        np.clip(prediction, 1e-7, 1 - 1e-7)
        / np.clip(1 - prediction, 1e-7, 1 - 1e-7)
    )
    low, high = -5.0, 5.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        shifted = 1.0 / (1.0 + np.exp(-(logits + middle)))
        if shifted.mean() < target_mean:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def apply_offsets(
    prediction: np.ndarray,
    game_type: np.ndarray,
    offsets: dict[str, float],
) -> np.ndarray:
    logits = np.log(
        np.clip(prediction, 1e-7, 1 - 1e-7)
        / np.clip(1 - prediction, 1e-7, 1 - 1e-7)
    )
    row_offsets = np.full(len(prediction), offsets["__overall__"], dtype=np.float64)
    for group in ("R", "F"):
        row_offsets[game_type == group] = offsets[group]
    return 1.0 / (1.0 + np.exp(-(logits + row_offsets)))


def forecast_validation_priors(full: pd.DataFrame) -> dict[str, float]:
    history = full.loc[full.season.lt(VALID_YEAR)]
    result = {}
    for group, weight in [("R", 0.75), ("F", 0.0)]:
        rates = (
            history.loc[history.game_type.eq(group)]
            .groupby("season", observed=True)
            .control_success.mean()
        )
        result[group] = (
            forecast_futures_prior(rates, VALID_YEAR)
            if group == "F"
            else forecast_blend(rates, VALID_YEAR, weight)
        )
    rates = history.groupby("season", observed=True).control_success.mean()
    result["__overall__"] = forecast_blend(rates, VALID_YEAR, 0.75)
    return result


def train_catboost_validation(
    features: pd.DataFrame,
    names: list[str],
    cat_indices: list[int],
    full: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    train_pool = Pool(
        features.loc[train_mask],
        label=full.loc[train_mask, "control_success"].to_numpy(np.int8),
        weight=weights[train_mask],
        cat_features=cat_indices,
        feature_names=names,
    )
    valid_pool = Pool(
        features.loc[valid_mask],
        label=full.loc[valid_mask, "control_success"].to_numpy(np.int8),
        cat_features=cat_indices,
        feature_names=names,
    )
    model = CatBoostClassifier(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="BrierScore",
        l2_leaf_reg=20,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        rsm=0.72,
        one_hot_max_size=20,
        max_ctr_complexity=1,
        has_time=True,
        random_seed=426,
        thread_count=6,
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=70)
    prediction = model.predict_proba(valid_pool)[:, 1]
    iterations = max(50, model.get_best_iteration() + 1)
    print(f"CatBoost validation: iterations={iterations}", flush=True)
    return prediction, iterations


def lgb_params(leaves: int, seed: int) -> dict:
    return {
        "objective": "binary",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": leaves,
        "min_data_in_leaf": 2000,
        "lambda_l2": 50,
        "feature_fraction": 0.72,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": 6,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
    }


def train_lgb_validation(
    matrix: np.ndarray,
    names: list[str],
    cat_indices: list[int],
    full: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    leaves: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    train_set = lgb.Dataset(
        matrix[train_mask],
        label=full.loc[train_mask, "control_success"].to_numpy(np.int8),
        weight=weights[train_mask],
        categorical_feature=cat_indices,
        feature_name=names,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        matrix[valid_mask],
        label=full.loc[valid_mask, "control_success"].to_numpy(np.int8),
        categorical_feature=cat_indices,
        feature_name=names,
        reference=train_set,
        free_raw_data=True,
    )
    model = lgb.train(
        lgb_params(leaves, seed),
        train_set,
        num_boost_round=900,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)],
    )
    prediction = model.predict(matrix[valid_mask], num_iteration=model.best_iteration)
    iterations = max(50, int(model.best_iteration))
    print(f"LightGBM validation: leaves={leaves} iterations={iterations}", flush=True)
    return prediction, iterations


def train_final_catboost(
    features: pd.DataFrame,
    names: list[str],
    cat_indices: list[int],
    labels: np.ndarray,
    weights: np.ndarray,
    iterations: int,
) -> tuple[list[dict], list[CatBoostClassifier]]:
    pool = Pool(
        features,
        label=labels,
        weight=weights,
        cat_features=cat_indices,
        feature_names=names,
    )
    members = []
    models = []
    for seed in CATBOOST_SEEDS:
        model = CatBoostClassifier(
            iterations=iterations,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            l2_leaf_reg=20,
            random_strength=0.5,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.72,
            one_hot_max_size=20,
            max_ctr_complexity=1,
            has_time=True,
            random_seed=seed,
            thread_count=6,
            allow_writing_files=False,
            verbose=50,
        )
        model.fit(pool)
        filename = f"catboost_seed{seed}.cbm"
        model.save_model(str(MODEL_DIR / filename))
        members.append(
            {
                "file": filename,
                "component": "catboost",
                "model_type": "classifier",
                "feature_set": "base",
                "weight": 0.5,
            }
        )
        models.append(model)
    return members, models


def train_final_catboost_decay(
    features: pd.DataFrame,
    names: list[str],
    cat_indices: list[int],
    labels: np.ndarray,
    weights: np.ndarray,
    classifier_iterations: int,
    regressor_iterations: int,
) -> tuple[list[dict], list[CatBoostClassifier], list[CatBoostRegressor]]:
    pool = Pool(
        features,
        label=labels,
        weight=weights,
        cat_features=cat_indices,
        feature_names=names,
    )
    members: list[dict] = []
    classifier_models: list[CatBoostClassifier] = []
    regressor_models: list[CatBoostRegressor] = []
    common = {
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 20,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.72,
        "one_hot_max_size": 20,
        "max_ctr_complexity": 1,
        "has_time": True,
        "thread_count": 6,
        "allow_writing_files": False,
        "verbose": 50,
    }
    for seed in CATBOOST_SEEDS:
        classifier = CatBoostClassifier(
            iterations=classifier_iterations,
            loss_function="Logloss",
            random_seed=seed,
            **common,
        )
        classifier.fit(pool)
        filename = f"catboost_decay_classifier_seed{seed}.cbm"
        classifier.save_model(str(MODEL_DIR / filename))
        members.append(
            {
                "file": filename,
                "component": "catboost_decay_classifier",
                "model_type": "classifier",
                "feature_set": "decay",
                "weight": 0.5,
            }
        )
        classifier_models.append(classifier)

        regressor = CatBoostRegressor(
            iterations=regressor_iterations,
            loss_function="RMSE",
            random_seed=seed,
            **common,
        )
        regressor.fit(pool)
        filename = f"catboost_decay_regressor_seed{seed}.cbm"
        regressor.save_model(str(MODEL_DIR / filename))
        members.append(
            {
                "file": filename,
                "component": "catboost_decay_regressor",
                "model_type": "regressor",
                "feature_set": "decay",
                "weight": 0.5,
            }
        )
        regressor_models.append(regressor)
    return members, classifier_models, regressor_models


def train_final_lightgbm(
    matrix: np.ndarray,
    names: list[str],
    cat_indices: list[int],
    labels: np.ndarray,
    weights: np.ndarray,
    rounds: dict[int, int],
) -> tuple[list[dict], list[lgb.Booster]]:
    train_set = lgb.Dataset(
        matrix,
        label=labels,
        weight=weights,
        categorical_feature=cat_indices,
        feature_name=names,
        free_raw_data=False,
    )
    configurations = [
        (15, 202, "lgb15", 1.0),
        (31, 303, "lgb31", 0.5),
        (31, 991, "lgb31", 0.5),
    ]
    members = []
    models = []
    for leaves, seed, component, weight in configurations:
        model = lgb.train(
            lgb_params(leaves, seed),
            train_set,
            num_boost_round=rounds[leaves],
            callbacks=[lgb.log_evaluation(50)],
        )
        filename = f"lightgbm_l{leaves}_seed{seed}.txt"
        model.save_model(str(MODEL_DIR / filename))
        members.append(
            {
                "file": filename,
                "component": component,
                "feature_set": "base",
                "weight": weight,
            }
        )
        models.append(model)
    return members, models


def train_final_lightgbm_decay(
    matrix: np.ndarray,
    names: list[str],
    cat_indices: list[int],
    labels: np.ndarray,
    weights: np.ndarray,
    iterations: int,
) -> tuple[list[dict], list[lgb.Booster]]:
    train_set = lgb.Dataset(
        matrix,
        label=labels,
        weight=weights,
        categorical_feature=cat_indices,
        feature_name=names,
        free_raw_data=False,
    )
    members = []
    models = []
    for seed in LGB_DECAY_SEEDS:
        model = lgb.train(
            lgb_params(31, seed),
            train_set,
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(50)],
        )
        filename = f"lightgbm_decay_l31_seed{seed}.txt"
        model.save_model(str(MODEL_DIR / filename))
        members.append(
            {
                "file": filename,
                "component": "lgb_decay",
                "feature_set": "decay",
                "weight": 0.5,
            }
        )
        models.append(model)
    return members, models


def main() -> None:
    global MODEL_DIR
    started = time.time()
    solution = load_solution_module()
    published_model_dir = MODEL_DIR
    staging_model_dir = ROOT / "submit/model.build"
    previous_model_dir = ROOT / "submit/model.previous"
    if staging_model_dir.exists() or previous_model_dir.exists():
        raise RuntimeError("stale model.build or model.previous directory exists")
    if not published_model_dir.is_dir():
        raise RuntimeError("published submit/model directory is missing")
    staging_model_dir.mkdir()
    MODEL_DIR = staging_model_dir

    print("Loading train.csv...", flush=True)
    full = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    required_columns = [column for column in full.columns if column != "control_success"]
    if len(full) != 1_475_092 or not full.row_id.is_unique:
        raise RuntimeError("unexpected training data shape or duplicate row_id")
    meta = make_base_meta(full, solution)
    meta["required_columns"] = required_columns
    print("2025 priors", json.dumps(meta["priors"]), flush=True)

    print("Building base features...", flush=True)
    cat_base, cat_base_names, cat_indices = solution.build_base_features(
        full, meta, include_ids=True
    )
    lgb_base, lgb_base_names, lgb_cat_indices = solution.build_base_features(
        full, meta, include_ids=False
    )

    print("Building current-season decompositions...", flush=True)
    pitcher_matrix, pitcher_names = entity_season_features_training(
        full,
        "pitcher_id",
        "asof_pitcher_n",
        solution.PITCHER_RATE_COLUMNS,
        "p",
    )
    batter_matrix, batter_names = entity_season_features_training(
        full,
        "batter_id",
        "asof_batter_n",
        solution.BATTER_RATE_COLUMNS,
        "b",
    )
    season_matrix = np.column_stack([pitcher_matrix, batter_matrix]).astype(np.float32)
    season_names = pitcher_names + batter_names
    del pitcher_matrix, batter_matrix
    meta["entity_history"] = {
        "p": fit_entity_history(
            full,
            "pitcher_id",
            "asof_pitcher_n",
            solution.PITCHER_RATE_COLUMNS,
        ),
        "b": fit_entity_history(
            full,
            "batter_id",
            "asof_batter_n",
            solution.BATTER_RATE_COLUMNS,
        ),
    }

    print("Building time-safe target encodings...", flush=True)
    te_matrix, te_names = temporal_target_encodings_training(full)
    meta["target_encodings"] = fit_target_encoding_meta(full)
    meta["te_prior"] = float(full.control_success.mean())

    print("Building time-decayed interaction encodings...", flush=True)
    rich_decay_matrix, rich_decay_names = temporal_rich_decay_encodings_training(
        full, solution
    )
    rich_encoding_meta, rich_te_prior = fit_rich_decay_meta(full, solution)
    meta["rich_target_encodings"] = rich_encoding_meta
    meta["rich_te_prior"] = rich_te_prior
    meta["rich_te_half_life"] = 2.0

    print("Loading and aggregating Trackman...", flush=True)
    trackman = pd.read_csv(TRACKMAN_PATH, encoding="utf-8-sig", low_memory=False)
    main_agg, main_hands = main_pitch_fingerprints(full)
    track_agg, track_hands = trackman_pitch_fingerprints(trackman)
    mapping = resolve_pitchers(
        main_agg, main_hands, track_agg, track_hands, VALID_YEAR
    )
    profiles = build_trackman_profiles(trackman, list(range(2019, TARGET_YEAR)))
    trackman_matrix, trackman_names = attach_trackman_training(full, mapping, profiles)
    target_profile = profiles[VALID_YEAR]
    profile_names = target_profile.columns.tolist()
    meta["trackman"] = fit_trackman_meta(mapping, target_profile, profile_names)
    del trackman, main_agg, track_agg, profiles
    strict_count = int(
        (
            mapping.tm_map_score.lt(0.9)
            & mapping.tm_map_margin.gt(0.25)
            & mapping.tm_map_reciprocal.gt(0)
            & mapping.tm_map_overlap_years.ge(2)
        ).sum()
    )
    print(f"Trackman mappings={len(mapping)} strict={strict_count}", flush=True)

    print("Assembling CatBoost matrix...", flush=True)
    cat_matrix = np.column_stack([cat_base, season_matrix, te_matrix]).astype(np.float32)
    cat_names = cat_base_names + season_names + te_names
    cat_features = catboost_frame(cat_matrix, cat_names, cat_indices)
    cat_decay_matrix = np.column_stack([cat_matrix, rich_decay_matrix]).astype(
        np.float32
    )
    cat_decay_names = cat_names + rich_decay_names
    cat_decay_features = catboost_frame(
        cat_decay_matrix, cat_decay_names, cat_indices
    )
    del cat_matrix, cat_decay_matrix, cat_base

    print("Assembling LightGBM matrix...", flush=True)
    lgb_matrix = np.column_stack(
        [lgb_base, season_matrix, te_matrix, trackman_matrix]
    ).astype(np.float32)
    lgb_names = lgb_base_names + season_names + te_names + trackman_names
    lgb_decay_matrix = np.column_stack([lgb_matrix, rich_decay_matrix]).astype(
        np.float32
    )
    lgb_decay_names = lgb_names + rich_decay_names
    del lgb_base, trackman_matrix, rich_decay_matrix

    meta["cat_feature_names"] = cat_names
    meta["cat_decay_feature_names"] = cat_decay_names
    meta["cat_indices"] = cat_indices
    meta["lgb_feature_names"] = lgb_names
    meta["lgb_decay_feature_names"] = lgb_decay_names
    meta["lgb_cat_indices"] = lgb_cat_indices

    valid_mask = full.season.eq(VALID_YEAR).to_numpy()
    labels = full.control_success.to_numpy(np.int8)
    weights = np.power(
        0.5, (VALID_YEAR - full.season.to_numpy(np.float64)) / 2.0
    ).astype(np.float32)
    game_type_valid = full.loc[valid_mask, "game_type"].astype(str).to_numpy()
    final_cat_rounds = FINAL_ROUNDS["catboost"]
    final_cat_decay_classifier_rounds = FINAL_ROUNDS[
        "catboost_decay_classifier"
    ]
    final_cat_decay_regressor_rounds = FINAL_ROUNDS[
        "catboost_decay_regressor"
    ]
    final_lgb_rounds = {15: FINAL_ROUNDS["lgb15"], 31: FINAL_ROUNDS["lgb31"]}
    final_lgb_decay_rounds = FINAL_ROUNDS["lgb_decay"]
    print("Frozen rolling OOF validation", json.dumps(OOF_VALIDATION), flush=True)
    print(
        "Training final models: "
        f"cat={final_cat_rounds} "
        f"cat_decay(classifier/regressor)="
        f"{final_cat_decay_classifier_rounds}/{final_cat_decay_regressor_rounds} "
        f"lgb={final_lgb_rounds} lgb_decay={final_lgb_decay_rounds}",
        flush=True,
    )
    cat_members, cat_models = train_final_catboost(
        cat_features,
        cat_names,
        cat_indices,
        labels,
        weights,
        final_cat_rounds,
    )
    (
        cat_decay_members,
        cat_decay_classifier_models,
        cat_decay_regressor_models,
    ) = train_final_catboost_decay(
        cat_decay_features,
        cat_decay_names,
        cat_indices,
        labels,
        weights,
        final_cat_decay_classifier_rounds,
        final_cat_decay_regressor_rounds,
    )
    cat_members.extend(cat_decay_members)
    lgb_members, lgb_models = train_final_lightgbm(
        lgb_matrix,
        lgb_names,
        lgb_cat_indices,
        labels,
        weights,
        final_lgb_rounds,
    )
    lgb_decay_members, lgb_decay_models = train_final_lightgbm_decay(
        lgb_decay_matrix,
        lgb_decay_names,
        lgb_cat_indices,
        labels,
        weights,
        final_lgb_decay_rounds,
    )
    lgb_members.extend(lgb_decay_members)
    lgb_models.extend(lgb_decay_models)

    print("Freezing row-independent 2025 calibration...", flush=True)
    reference_cat_pool = Pool(
        cat_features.loc[valid_mask],
        cat_features=cat_indices,
        feature_names=cat_names,
    )
    reference_cat = np.mean(
        np.column_stack(
            [model.predict_proba(reference_cat_pool)[:, 1] for model in cat_models]
        ),
        axis=1,
    )
    reference_cat_decay_pool = Pool(
        cat_decay_features.loc[valid_mask],
        cat_features=cat_indices,
        feature_names=cat_decay_names,
    )
    reference_cat_decay_classifier = np.mean(
        np.column_stack(
            [
                model.predict_proba(reference_cat_decay_pool)[:, 1]
                for model in cat_decay_classifier_models
            ]
        ),
        axis=1,
    )
    reference_cat_decay_regressor = np.clip(
        np.mean(
            np.column_stack(
                [
                    model.predict(reference_cat_decay_pool)
                    for model in cat_decay_regressor_models
                ]
            ),
            axis=1,
        ),
        1e-5,
        1.0 - 1e-5,
    )
    reference_lgb: dict[str, np.ndarray] = {
        "lgb15": np.zeros(valid_mask.sum(), dtype=np.float64),
        "lgb31": np.zeros(valid_mask.sum(), dtype=np.float64),
        "lgb_decay": np.zeros(valid_mask.sum(), dtype=np.float64),
    }
    reference_lgb_weights = {"lgb15": 0.0, "lgb31": 0.0, "lgb_decay": 0.0}
    for member, model in zip(lgb_members, lgb_models):
        component = str(member["component"])
        member_weight = float(member["weight"])
        reference_matrix = (
            lgb_decay_matrix if member["feature_set"] == "decay" else lgb_matrix
        )
        reference_lgb[component] += member_weight * model.predict(
            reference_matrix[valid_mask]
        )
        reference_lgb_weights[component] += member_weight
    for component in reference_lgb:
        reference_lgb[component] /= reference_lgb_weights[component]

    priors = meta["priors"]
    reference_components = {
        "catboost": reference_cat,
        "catboost_decay_classifier": reference_cat_decay_classifier,
        "catboost_decay_regressor": reference_cat_decay_regressor,
        "lgb15": reference_lgb["lgb15"],
        "lgb31": reference_lgb["lgb31"],
        "lgb_decay": reference_lgb["lgb_decay"],
    }
    component_offsets: dict[str, dict[str, float]] = {}
    for component, prediction in reference_components.items():
        component_offsets[component] = {
            "__overall__": solve_offset(prediction, priors["__overall__"]),
            "R": solve_offset(prediction[game_type_valid == "R"], priors["R"]),
            "F": solve_offset(prediction[game_type_valid == "F"], priors["F"]),
        }
    print(
        "Calibration",
        json.dumps(
            {
                "reference_component_means": {
                    name: float(prediction.mean())
                    for name, prediction in reference_components.items()
                },
                "component_offsets": component_offsets,
                "target_priors": priors,
            }
        ),
        flush=True,
    )

    meta["catboost_models"] = cat_members
    meta["lightgbm_models"] = lgb_members
    meta["stack_coefficients"] = STACK_COEFFICIENTS
    meta["component_calibration_offsets"] = component_offsets
    with (MODEL_DIR / "meta.pkl").open("wb") as file:
        pickle.dump(meta, file, protocol=5)

    manifest = {
        "validation": OOF_VALIDATION,
        "iterations": {
            "catboost": {
                "base_classifier": final_cat_rounds,
                "decay_classifier": final_cat_decay_classifier_rounds,
                "decay_regressor": final_cat_decay_regressor_rounds,
            },
            "lightgbm": {
                "base": final_lgb_rounds,
                "decay_l31": final_lgb_decay_rounds,
            },
        },
        "stack_coefficients": STACK_COEFFICIENTS,
        "priors": priors,
        "component_calibration_offsets": component_offsets,
        "feature_counts": {
            "catboost": len(cat_names),
            "catboost_decay": len(cat_decay_names),
            "lightgbm": len(lgb_names),
            "lightgbm_decay": len(lgb_decay_names),
        },
        "models": {"catboost": cat_members, "lightgbm": lgb_members},
        "trackman_mappings": len(mapping),
        "training_seconds": round(time.time() - started, 1),
    }
    with (MODEL_DIR / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    published_model_dir.rename(previous_model_dir)
    staging_model_dir.rename(published_model_dir)
    MODEL_DIR = published_model_dir
    print("Build complete", json.dumps(manifest, ensure_ascii=False), flush=True)
    print(f"Previous model retained at {previous_model_dir}", flush=True)


if __name__ == "__main__":
    main()

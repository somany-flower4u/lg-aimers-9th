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
from catboost import CatBoostClassifier, Pool


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
TE_DEFINITIONS = [
    ("pte_game_type", ["pitcher_id", "game_type"], 100.0),
    ("pte_batter_hand", ["pitcher_id", "batter_hand"], 120.0),
    ("pte_count", ["pitcher_id", "balls_before", "strikes_before"], 180.0),
    ("bte_pitcher_hand", ["batter_id", "pitcher_hand"], 250.0),
    ("pair_te", ["pitcher_id", "batter_id"], 80.0),
]


def load_solution_module():
    path = ROOT / "submit/script.py"
    specification = importlib.util.spec_from_file_location("submission_solution", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
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
        "F": forecast_blend(futures, TARGET_YEAR, 0.90),
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


def main_pitch_fingerprints(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = train[
        [
            "pitcher_id",
            "pitcher_hand",
            "season",
            "asof_pitcher_n",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].copy()
    for group in PITCH_GROUPS[:3]:
        counts = np.rint(
            np.nan_to_num(
                frame[f"asof_pitcher_{group}_rate"].to_numpy(np.float64)
                * frame.asof_pitcher_n.to_numpy(np.float64),
                nan=0.0,
            )
        )
        count_series = pd.Series(counts, index=frame.index)
        frame[group] = count_series.groupby(frame.pitcher_id, sort=False).shift(-1) - counts
    next_n = frame.asof_pitcher_n.groupby(frame.pitcher_id, sort=False).shift(-1)
    deltas = frame[PITCH_GROUPS[:3]]
    valid = (
        next_n.eq(frame.asof_pitcher_n + 1)
        & deltas.sum(axis=1).eq(1)
        & deltas.isin([0, 1]).all(axis=1)
    )
    aggregates = (
        frame.loc[valid]
        .groupby(["pitcher_id", "season"], observed=True)
        .agg(
            n=("pitcher_id", "size"),
            fastball=("fastball", "sum"),
            breaking=("breaking", "sum"),
            offspeed=("offspeed", "sum"),
        )
        .reset_index()
    )
    hands = (
        frame.groupby("pitcher_id", observed=True).pitcher_hand.first().map({1: "Left", 2: "Right"})
    )
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
    hands = valid.groupby("pitcher_trackman_id", observed=True).pitcher_hand.first()
    return aggregates, hands


def resolve_pitchers(
    main_agg: pd.DataFrame,
    main_hands: pd.Series,
    track_agg: pd.DataFrame,
    track_hands: pd.Series,
    cutoff: int,
) -> pd.DataFrame:
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

    best_j = np.argmin(scores, axis=1)
    best_score = scores[np.arange(len(main_ids)), best_j]
    second_score = np.partition(scores, 1, axis=1)[:, 1]
    best_i = np.argmin(scores, axis=0)
    rows = []
    for i, j in enumerate(best_j):
        if not np.isfinite(best_score[i]):
            continue
        mix_rms, count_shape, jaccard, overlap = details[(i, j)]
        rows.append(
            {
                "pitcher_id": int(main_ids[i]),
                "pitcher_trackman_id": int(track_ids[j]),
                "tm_map_score": float(best_score[i]),
                "tm_map_margin": float(second_score[i] - best_score[i]),
                "tm_map_reciprocal": float(best_i[j] == i),
                "tm_map_mix_rms": float(mix_rms),
                "tm_map_count_shape": float(count_shape),
                "tm_map_jaccard": float(jaccard),
                "tm_map_overlap_years": float(overlap),
            }
        )
    return pd.DataFrame(rows)


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
        result[group] = forecast_blend(rates, VALID_YEAR, weight)
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
        eval_metric="Logloss",
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
        "metric": "binary_logloss",
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
    for seed in (426, 990):
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
        members.append({"file": filename, "weight": 0.5})
        models.append(model)
    return members, models


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
        (15, 202, 0.25),
        (31, 303, 0.375),
        (31, 991, 0.375),
    ]
    members = []
    models = []
    for leaves, seed, weight in configurations:
        model = lgb.train(
            lgb_params(leaves, seed),
            train_set,
            num_boost_round=rounds[leaves],
            callbacks=[lgb.log_evaluation(50)],
        )
        filename = f"lightgbm_l{leaves}_seed{seed}.txt"
        model.save_model(str(MODEL_DIR / filename))
        members.append({"file": filename, "weight": weight})
        models.append(model)
    return members, models


def main() -> None:
    started = time.time()
    solution = load_solution_module()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path in MODEL_DIR.iterdir():
        if path.is_file():
            path.unlink()

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
    del cat_matrix, cat_base

    print("Assembling LightGBM matrix...", flush=True)
    lgb_matrix = np.column_stack(
        [lgb_base, season_matrix, te_matrix, trackman_matrix]
    ).astype(np.float32)
    lgb_names = lgb_base_names + season_names + te_names + trackman_names
    del lgb_base, trackman_matrix

    meta["cat_feature_names"] = cat_names
    meta["cat_indices"] = cat_indices
    meta["lgb_feature_names"] = lgb_names
    meta["lgb_cat_indices"] = lgb_cat_indices

    train_mask = full.season.lt(VALID_YEAR).to_numpy()
    valid_mask = full.season.eq(VALID_YEAR).to_numpy()
    labels = full.control_success.to_numpy(np.int8)
    weights = np.power(
        0.5, (VALID_YEAR - full.season.to_numpy(np.float64)) / 2.0
    ).astype(np.float32)
    game_type_valid = full.loc[valid_mask, "game_type"].astype(str).to_numpy()
    y_valid = labels[valid_mask]

    print("Selecting iterations on the 2024 holdout...", flush=True)
    cat_valid, cat_rounds = train_catboost_validation(
        cat_features,
        cat_names,
        cat_indices,
        full,
        train_mask,
        valid_mask,
        weights,
    )
    lgb15_valid, lgb15_rounds = train_lgb_validation(
        lgb_matrix,
        lgb_names,
        lgb_cat_indices,
        full,
        train_mask,
        valid_mask,
        weights,
        leaves=15,
        seed=202,
    )
    lgb31_valid, lgb31_rounds = train_lgb_validation(
        lgb_matrix,
        lgb_names,
        lgb_cat_indices,
        full,
        train_mask,
        valid_mask,
        weights,
        leaves=31,
        seed=303,
    )
    lgb_valid = 0.25 * lgb15_valid + 0.75 * lgb31_valid
    validation_priors = forecast_validation_priors(full)
    best = None
    for cat_weight in np.linspace(0.40, 0.85, 10):
        raw = cat_weight * cat_valid + (1.0 - cat_weight) * lgb_valid
        offsets = {
            "__overall__": solve_offset(raw, validation_priors["__overall__"]),
            "R": solve_offset(raw[game_type_valid == "R"], validation_priors["R"]),
            "F": solve_offset(raw[game_type_valid == "F"], validation_priors["F"]),
        }
        calibrated = apply_offsets(raw, game_type_valid, offsets)
        score = logloss(y_valid, calibrated)
        candidate = (score, float(cat_weight), offsets, calibrated)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    validation_score, selected_cat_weight, validation_offsets, validation_prediction = best
    print(
        "Validation result",
        json.dumps(
            {
                "logloss": validation_score,
                "auc": auc(y_valid, validation_prediction),
                "cat_weight": selected_cat_weight,
                "offsets": validation_offsets,
                "priors": validation_priors,
            }
        ),
        flush=True,
    )

    final_cat_rounds = max(50, int(round(cat_rounds * 1.05)))
    final_lgb_rounds = {
        15: max(50, int(round(lgb15_rounds * 1.05))),
        31: max(50, int(round(lgb31_rounds * 1.05))),
    }
    print(
        f"Training final models: cat={final_cat_rounds} lgb={final_lgb_rounds}",
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
    lgb_members, lgb_models = train_final_lightgbm(
        lgb_matrix,
        lgb_names,
        lgb_cat_indices,
        labels,
        weights,
        final_lgb_rounds,
    )

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
    reference_lgb = np.zeros(valid_mask.sum(), dtype=np.float64)
    for member, model in zip(lgb_members, lgb_models):
        reference_lgb += float(member["weight"]) * model.predict(lgb_matrix[valid_mask])
    reference_lgb /= sum(float(member["weight"]) for member in lgb_members)
    reference_raw = (
        selected_cat_weight * reference_cat
        + (1.0 - selected_cat_weight) * reference_lgb
    )
    priors = meta["priors"]
    reference_offsets = {
        "__overall__": solve_offset(reference_raw, priors["__overall__"]),
        "R": solve_offset(reference_raw[game_type_valid == "R"], priors["R"]),
        "F": solve_offset(reference_raw[game_type_valid == "F"], priors["F"]),
    }
    print(
        "Calibration",
        json.dumps(
            {
                "reference_raw_mean": float(reference_raw.mean()),
                "offsets": reference_offsets,
                "target_priors": priors,
            }
        ),
        flush=True,
    )

    meta["catboost_models"] = cat_members
    meta["lightgbm_models"] = lgb_members
    meta["blend_weights"] = {
        "catboost": selected_cat_weight,
        "lightgbm": 1.0 - selected_cat_weight,
    }
    meta["calibration_offsets"] = reference_offsets
    with (MODEL_DIR / "meta.pkl").open("wb") as file:
        pickle.dump(meta, file, protocol=5)

    manifest = {
        "validation": {
            "year": VALID_YEAR,
            "logloss": validation_score,
            "auc": auc(y_valid, validation_prediction),
            "catboost_weight": selected_cat_weight,
        },
        "iterations": {
            "catboost": final_cat_rounds,
            "lightgbm": final_lgb_rounds,
        },
        "priors": priors,
        "calibration_offsets": reference_offsets,
        "feature_counts": {"catboost": len(cat_names), "lightgbm": len(lgb_names)},
        "models": {"catboost": cat_members, "lightgbm": lgb_members},
        "trackman_mappings": len(mapping),
        "training_seconds": round(time.time() - started, 1),
    }
    with (MODEL_DIR / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print("Build complete", json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

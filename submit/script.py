"""LG Aimers 9 Phase 2 inference entry point.

Every prediction uses one evaluation row plus artifacts frozen from the
2019-2024 training data.  The code never aggregates, sorts, rolls, or target
encodes evaluation rows.  Trackman's raw CSV is not read during inference.
"""
from __future__ import annotations

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
from catboost import CatBoostClassifier, Pool


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

TE_SPECS = [
    ("pte_game_type", ["pitcher_id", "game_type"]),
    ("pte_batter_hand", ["pitcher_id", "batter_hand"]),
    ("pte_count", ["pitcher_id", "balls_before", "strikes_before"]),
    ("bte_pitcher_hand", ["batter_id", "pitcher_hand"]),
    ("pair_te", ["pitcher_id", "batter_id"]),
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


def _catboost_frame(matrix: np.ndarray, names: list[str], cat_indices: list[int]) -> pd.DataFrame:
    result = pd.DataFrame(matrix, columns=names)
    for index in cat_indices:
        column = names[index]
        result[column] = result[column].fillna(-1).astype(np.int32)
    return result


def predict(frame: pd.DataFrame, meta: dict) -> np.ndarray:
    started = time.perf_counter()
    season_values = frame["season"].dropna().astype(int).unique().tolist()
    if season_values != [int(meta["target_year"])]:
        raise ValueError(
            f"expected only season={meta['target_year']}, received {season_values}"
        )

    season_matrix, season_names = build_season_features(frame, meta)
    te_matrix, te_names = build_te_features(frame, meta)

    cat_base, cat_base_names, cat_indices = build_base_features(frame, meta, include_ids=True)
    cat_matrix = np.column_stack([cat_base, season_matrix, te_matrix]).astype(np.float32)
    cat_names = cat_base_names + season_names + te_names
    if cat_names != meta["cat_feature_names"] or cat_indices != meta["cat_indices"]:
        raise RuntimeError("CatBoost feature layout does not match training")
    cat_frame = _catboost_frame(cat_matrix, cat_names, cat_indices)
    cat_pool = Pool(cat_frame, cat_features=cat_indices, feature_names=cat_names)
    cat_prediction = np.zeros(len(frame), dtype=np.float64)
    cat_weight = 0.0
    for member in meta["catboost_models"]:
        model = CatBoostClassifier()
        model.load_model(str(MODEL_DIR / member["file"]))
        cat_prediction += float(member["weight"]) * model.predict_proba(cat_pool)[:, 1]
        cat_weight += float(member["weight"])
    cat_prediction /= cat_weight
    del cat_pool, cat_frame, cat_matrix, cat_base

    lgb_base, lgb_base_names, lgb_cat_indices = build_base_features(
        frame, meta, include_ids=False
    )
    trackman_matrix, trackman_names = build_trackman_features(frame, meta)
    lgb_matrix = np.column_stack(
        [lgb_base, season_matrix, te_matrix, trackman_matrix]
    ).astype(np.float32)
    lgb_names = lgb_base_names + season_names + te_names + trackman_names
    if lgb_names != meta["lgb_feature_names"] or lgb_cat_indices != meta["lgb_cat_indices"]:
        raise RuntimeError("LightGBM feature layout does not match training")
    lgb_prediction = np.zeros(len(frame), dtype=np.float64)
    lgb_weight = 0.0
    for member in meta["lightgbm_models"]:
        model = lgb.Booster(model_file=str(MODEL_DIR / member["file"]))
        lgb_prediction += float(member["weight"]) * model.predict(lgb_matrix)
        lgb_weight += float(member["weight"])
    lgb_prediction /= lgb_weight

    raw = (
        float(meta["blend_weights"]["catboost"]) * cat_prediction
        + float(meta["blend_weights"]["lightgbm"]) * lgb_prediction
    )
    logits = logit(raw)
    game_type = frame["game_type"].astype(str).to_numpy()
    overall_offset = float(meta["calibration_offsets"]["__overall__"])
    offsets = np.full(len(frame), overall_offset, dtype=np.float64)
    for group, offset in meta["calibration_offsets"].items():
        if group != "__overall__":
            offsets[game_type == group] = float(offset)
    prediction = np.clip(sigmoid(logits + offsets), EPS, 1.0 - EPS)
    if not np.isfinite(prediction).all():
        raise RuntimeError("model produced non-finite probabilities")
    print(
        f"Inference complete: rows={len(frame)} features(cat/lgb)="
        f"{len(cat_names)}/{len(lgb_names)} seconds={time.perf_counter()-started:.2f}"
    )
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
    print(
        f"Saved {OUT_PATH}: rows={len(submission)} mean={ordered.mean():.6f} "
        f"total_seconds={time.perf_counter()-total_started:.2f}"
    )


if __name__ == "__main__":
    main()

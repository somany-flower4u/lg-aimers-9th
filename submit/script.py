"""LG Aimers Phase 2 -- inference entry point.

GENERATED FILE (build_submit.py). Every prediction is a function of that row
alone: the model is a frozen LightGBM blend and the season intercept was fixed
at training time, so no statistic of the evaluation set is ever computed here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb



# ---------- from src/common.py ----------

"""Shared utilities: metric, logit helpers, data loading."""


ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))









# ---------- from src/features.py ----------

"""Feature construction.

Hard constraint from the rules: every feature must be computable from a SINGLE
row plus static tables fitted on train.csv / trackman_history.csv.  Nothing here
may look at another test row, at the row ordering, or at anything determined
after the pitch is thrown.

``fit_meta`` is fitted once on the training window; ``build_features`` is the
one and only transform, used identically for training, backtesting and the
submitted inference script.
"""



# asof columns whose league level slides from season to season
DRIFT_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]

RAW_NUM = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand",
    "asof_pitcher_n", "asof_batter_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

BASE_STATES = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]

DEFAULT_CFG = dict(
    relativize=True,     # league-adjusted asof columns
    derived=True,        # count / leverage / matchup engineering
    shrink=True,         # sample-size shrunk asof rates
    log5=True,           # pitcher x batter log-odds matchup
    ids=True,            # pitcher_id / batter_id as categoricals
    teams=True,          # team ids as categoricals
    trackman=False,      # trackman-derived pitcher profiles
)






_INT_KEYED = ("pitcher_map", "batter_map", "team_map")




def meta_from_json(d: dict) -> dict:
    """Inverse of :func:`meta_to_json` -- restores integer dictionary keys."""
    meta = dict(d)
    for k in _INT_KEYED:
        if k in meta:
            meta[k] = {int(a): int(b) for a, b in meta[k].items()}
    meta["season_means"] = {c: {int(s): float(v) for s, v in t.items()}
                            for c, t in meta["season_means"].items()}
    return meta


def _season_level(meta, col, season_arr):
    table = meta["season_means"][col]
    lut = pd.Series(table)
    return season_arr.map(lut).astype(np.float64).values


def build_features(df: pd.DataFrame, meta: dict):
    """Return (X, feature_names, categorical_names) -- pure row-wise transform."""
    cfg = meta["cfg"]
    out: dict[str, np.ndarray] = {}
    season = df["season"].astype(int)

    for c in RAW_NUM:
        out[c] = df[c].astype(np.float64).values

    # ---- categoricals (integer coded; -1 = unseen) ----
    cats: list[str] = []
    out["top_bottom_c"] = (df["top_bottom"].astype(str) == "T").astype(np.float64).values
    out["game_type_c"] = (df["game_type"].astype(str) == "R").astype(np.float64).values
    bs_map = {v: i for i, v in enumerate(BASE_STATES)}
    out["base_state_c"] = df["base_state"].astype(str).map(bs_map).fillna(-1).astype(np.float64).values
    cats += ["base_state_c"]
    if cfg["ids"]:
        out["pitcher_c"] = df["pitcher_id"].astype(int).map(meta["pitcher_map"]).fillna(-1).astype(np.float64).values
        out["batter_c"] = df["batter_id"].astype(int).map(meta["batter_map"]).fillna(-1).astype(np.float64).values
        cats += ["pitcher_c", "batter_c"]
    if cfg["teams"]:
        out["pteam_c"] = df["pitcher_team_id"].astype(int).map(meta["team_map"]).fillna(-1).astype(np.float64).values
        out["bteam_c"] = df["batter_team_id"].astype(int).map(meta["team_map"]).fillna(-1).astype(np.float64).values
        cats += ["pteam_c", "bteam_c"]

    # ---- league-relative asof values ----
    if cfg["relativize"]:
        for c in DRIFT_COLS:
            lvl = _season_level(meta, c, season)
            out["rel_" + c] = df[c].astype(np.float64).values - lvl

    b = df["balls_before"].astype(np.float64).values
    s = df["strikes_before"].astype(np.float64).values
    p_n = df["asof_pitcher_n"].astype(np.float64).values
    bt_n = df["asof_batter_n"].astype(np.float64).values

    if cfg["derived"]:
        out["count_diff"] = b - s
        out["count_c"] = b * 3.0 + s
        cats += ["count_c"]
        out["is_3ball"] = (b == 3).astype(np.float64)
        out["is_2strike"] = (s == 2).astype(np.float64)
        out["pitcher_ahead"] = (s > b).astype(np.float64)
        out["batter_ahead"] = (b > s).astype(np.float64)
        out["hand_match_c"] = (df["pitcher_hand"].astype(np.float64).values * 2.0
                               + df["batter_hand"].astype(np.float64).values)
        cats += ["hand_match_c"]
        out["same_hand"] = (df["pitcher_hand"].values == df["batter_hand"].values).astype(np.float64)

        sd = df["score_diff_pitcher_team"].astype(np.float64).values
        out["abs_score_diff"] = np.abs(sd)
        out["close_game"] = (np.abs(sd) <= 2).astype(np.float64)
        inn = df["inning"].astype(np.float64).values
        out["late"] = (inn >= 7).astype(np.float64)
        out["late_close"] = out["late"] * out["close_game"]
        out["blowout"] = (np.abs(sd) >= 5).astype(np.float64)
        out["log_li"] = np.log1p(df["li"].astype(np.float64).values)

        # the home team pitches in the top half, so top_bottom identifies the side
        is_home_p = out["top_bottom_c"]
        hwe = df["home_win_expectancy"].astype(np.float64).values
        out["pitcher_we"] = np.where(is_home_p > 0, hwe, 100.0 - hwe)
        out["we_margin"] = np.abs(hwe - 50.0)
        out["is_pitcher_home"] = is_home_p

        out["log_pitcher_n"] = np.log1p(p_n)
        out["log_batter_n"] = np.log1p(bt_n)
        out["pitcher_cold"] = (p_n == 0).astype(np.float64)
        out["batter_cold"] = (bt_n == 0).astype(np.float64)
        out["prev_game_missing"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype(np.float64).values

        # residual failure mode: neither middle nor reverse -> "way off the zone"
        suc = df["asof_pitcher_success_rate"].astype(np.float64).values
        mid = df["asof_pitcher_middle_rate"].astype(np.float64).values
        rev = df["asof_pitcher_reverse_rate"].astype(np.float64).values
        out["p_wayoff_rate"] = np.clip(1.0 - suc - mid - rev, -1.0, 1.0)

        # short-term form relative to career level
        for k in (1, 3, 5):
            col = f"asof_pitcher_prev{k}_game_success_rate"
            out[f"form{k}"] = df[col].astype(np.float64).values - suc
            mcol = f"asof_pitcher_prev{k}_game_middle_rate"
            out[f"form{k}_mid"] = df[mcol].astype(np.float64).values - mid
        out["form_slope"] = (df["asof_pitcher_prev1_game_success_rate"].astype(np.float64).values
                             - df["asof_pitcher_prev5_game_success_rate"].astype(np.float64).values)

        mix = df[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                  "asof_pitcher_offspeed_rate"]].astype(np.float64).values
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mix_entropy"] = -np.nansum(np.where(mix > 0, mix * np.log(mix + 1e-12), 0.0), axis=1)
        out["nonfastball_rate"] = 1.0 - df["asof_pitcher_fastball_rate"].astype(np.float64).values

    # ---- sample-size shrunk rates ----
    if cfg["shrink"]:
        for col, n, k in [("asof_pitcher_success_rate", p_n, 150.0),
                          ("asof_pitcher_middle_rate", p_n, 150.0),
                          ("asof_pitcher_reverse_rate", p_n, 150.0),
                          ("asof_batter_success_rate", bt_n, 300.0),
                          ("asof_batter_middle_rate", bt_n, 300.0)]:
            lvl = _season_level(meta, col, season)
            raw = df[col].astype(np.float64).values
            raw = np.where(np.isnan(raw), lvl, raw)
            out["shr_" + col] = (raw * n + lvl * k) / (n + k)

    # ---- log5 style matchup on the relativized scale ----
    if cfg["log5"]:
        lvl_p = _season_level(meta, "asof_pitcher_success_rate", season)
        lvl_b = _season_level(meta, "asof_batter_success_rate", season)
        pp = df["asof_pitcher_success_rate"].astype(np.float64).values
        bb = df["asof_batter_success_rate"].astype(np.float64).values
        pp = np.where(np.isnan(pp), lvl_p, pp)
        bb = np.where(np.isnan(bb), lvl_b, bb)
        pp = (pp * p_n + lvl_p * 150.0) / (p_n + 150.0)
        bb = (bb * bt_n + lvl_b * 300.0) / (bt_n + 300.0)
        base = 0.5 * (lvl_p + lvl_b)
        out["log5"] = logit(pp) + logit(bb) - logit(base)
        out["pb_gap"] = logit(pp) - logit(bb)

        lvl_pm = _season_level(meta, "asof_pitcher_middle_rate", season)
        lvl_bm = _season_level(meta, "asof_batter_middle_rate", season)
        pm = df["asof_pitcher_middle_rate"].astype(np.float64).values
        bm = df["asof_batter_middle_rate"].astype(np.float64).values
        pm = np.where(np.isnan(pm), lvl_pm, pm)
        bm = np.where(np.isnan(bm), lvl_bm, bm)
        pm = (pm * p_n + lvl_pm * 150.0) / (p_n + 150.0)
        bm = (bm * bt_n + lvl_bm * 300.0) / (bt_n + 300.0)
        out["log5_mid"] = logit(pm) + logit(bm) - logit(0.5 * (lvl_pm + lvl_bm))

    # ---- trackman pitcher profile (optional) ----
    if cfg.get("trackman") and "trackman" in meta:
        tm = meta["trackman"]
        cols = meta["trackman_cols"]
        pid = df["pitcher_id"].astype(int).values
        buf = np.full((len(df), len(cols)), np.nan)
        for i, c in enumerate(cols):
            lut = {k: v[c] for k, v in tm.items()}
            buf[:, i] = pd.Series(pid).map(lut).astype(np.float64).values
        for i, c in enumerate(cols):
            out["tm_" + c] = buf[:, i]

    names = list(out.keys())
    X = np.empty((len(df), len(names)), dtype=np.float32)
    for i, n in enumerate(names):
        X[:, i] = out[n]
    cat_idx = [names.index(c) for c in cats if c in names]
    return X, names, cat_idx


# ============================== inference ==============================

DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_PATH = "./output/submission.csv"


def load_meta(model_dir):
    with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
        return meta_from_json(json.load(f))


def predict(test, meta, model_dir):
    X, names, _ = build_features(test, meta)
    if names != meta["feature_names"]:
        raise RuntimeError("feature layout drifted from the trained model")
    gt = test["game_type"].astype(str).values
    total = np.zeros(len(test), dtype=np.float64)
    wsum = 0.0
    for grp in meta["groups"]:
        z = np.zeros(len(test), dtype=np.float64)
        gw = 0.0
        for item in grp["members"]:
            bst = lgb.Booster(model_file=os.path.join(model_dir, item["file"]))
            z += item["weight"] * bst.predict(X, raw_score=True)
            gw += item["weight"]
        z /= gw
        # The season intercept is a constant frozen at training time -- no
        # statistic of the evaluation set is computed anywhere in this file.
        off = grp["offsets"]
        for g in np.unique(gt):
            m = gt == g
            z[m] += off.get(g, off["__overall__"])
        total += grp["weight"] * z
        wsum += grp["weight"]
    return np.clip(sigmoid(total / wsum), 1e-6, 1.0 - 1e-6)


def main():
    print("Load model...")
    meta = load_meta(MODEL_DIR)
    n = sum(len(g["members"]) for g in meta["groups"])
    print(f" boosters={n} features={len(meta['feature_names'])} "
          f"target_season={meta['target_season']}")
    for g in meta["groups"]:
        print(f"  [{g['mode']}] w={g['weight']} offsets="
              f"{ {k: round(v, 4) for k, v in g['offsets'].items()} }")

    print("Load test data...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"),
                      encoding="utf-8-sig")
    print(f" test={len(test)} submission={len(sub)}")

    print("Inference...")
    p = predict(test, meta, MODEL_DIR)

    pred_map = dict(zip(test[ID_COL].tolist(), p.tolist()))
    fallback = float(meta["priors"]["__overall__"])
    values = [pred_map.get(rid, fallback) for rid in sub[ID_COL]]
    n_missing = sum(1 for rid in sub[ID_COL] if rid not in pred_map)
    if n_missing:
        print(f" WARNING: {n_missing} row_id(s) had no prediction; used prior")
    sub[TARGET_COL] = values

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)}, mean={np.mean(values):.4f})")


if __name__ == "__main__":
    main()

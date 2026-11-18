"""Evaluation entry point: production stack + frozen 4-class member.

Every feature is computed from the row itself plus lookup tables frozen at
training time.  No statistic is taken over test.csv.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_PATH = HERE / "output" / "submission.csv"
ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6
NEURAL_BLOCK = 4096


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, filename: str, model_dir: Path):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.MODEL_DIR = model_dir
    module.DATA_DIR = DATA_DIR
    return module


def main() -> None:
    started = time.perf_counter()
    prod = load_module("_prod", "_prod.py", HERE / "model")
    multi = load_module("_multi", "_multi.py", HERE / "model_multi")

    meta = pickle.loads((HERE / "model" / "meta.pkl").read_bytes())
    multi_meta = pickle.loads((HERE / "model_multi" / "meta.pkl").read_bytes())
    config = json.loads((HERE / "model_multi" / "blend.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise RuntimeError(
            f"unsupported multiclass blend schema: {config.get('schema_version')}"
        )

    scale = float(config["stack_scale"])
    meta["stack_coefficients"] = {
        name: float(value) * scale
        for name, value in meta["stack_coefficients"].items()
    }

    specialist = config.get("league_specialist")
    if specialist is not None:
        required = {"file", "sha256", "iterations", "league", "mode"}
        if not isinstance(specialist, dict) or not required.issubset(specialist):
            raise RuntimeError(f"incomplete league specialist config: {specialist!r}")
        if specialist["league"] not in ("R", "F"):
            raise RuntimeError(f"invalid league specialist league: {specialist['league']!r}")
        if specialist["mode"] not in ("replace", "blend", "off"):
            raise RuntimeError(f"invalid league specialist mode: {specialist['mode']!r}")
        if specialist["mode"] != "off":
            meta["league_specialist"] = specialist

    slot_policy = config.get("lgb_slot_policy", "half_and_half")
    if slot_policy == "decay_only":
        coefficients = meta["stack_coefficients"]
        coefficients["lgb_decay"] = (
            float(coefficients["lgb_decay"]) + float(coefficients["lgb31"])
        )
        coefficients["lgb31"] = 0.0
    elif slot_policy != "half_and_half":
        raise RuntimeError(f"unsupported LightGBM slot policy: {slot_policy!r}")

    candidate = json.loads(
        (HERE / "trackman_candidate.json").read_text(encoding="utf-8")
    )
    if candidate.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported TrackMan candidate schema: {candidate.get('schema_version')}"
        )
    gate = candidate.get("gate")
    offsets = candidate.get("lgb31_calibration_offsets")
    if not isinstance(gate, dict) or not isinstance(offsets, dict):
        raise RuntimeError("TrackMan candidate configuration is incomplete")
    if set(offsets) != {"__overall__", "R", "F"}:
        raise RuntimeError(
            f"invalid LGB31 calibration groups: {sorted(offsets)}"
        )
    meta["trackman_confidence_gate"] = gate
    meta["component_calibration_offsets"]["lgb31"] = {
        group: float(value) for group, value in offsets.items()
    }

    policy = candidate.get("level_policy")
    if not isinstance(policy, dict) or policy.get("name") != "P0_raw":
        raise RuntimeError(f"unsupported level policy: {policy!r}")
    if policy.get("action") != "zero_component_calibration_offsets":
        raise RuntimeError(f"unsupported level policy action: {policy!r}")
    meta["component_calibration_offsets"] = {
        component: {group: 0.0 for group in groups}
        for component, groups in meta["component_calibration_offsets"].items()
    }

    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig", low_memory=False)
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv",
                             encoding="utf-8-sig", low_memory=False)
    prod._validate_inputs(test, submission, meta)

    captured: dict = {}
    base = prod.predict(test, meta, capture=captured)

    matrix = multi.build_inference_matrix(test, multi_meta)
    members = config.get("multi_members")
    if not isinstance(members, list) or len(members) != 3:
        raise RuntimeError("multiclass blend must contain exactly three members")
    member_predictions: list[np.ndarray] = []
    member_weights: list[float] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict) or set(member) != {
            "file",
            "iterations",
            "weight",
            "sha256",
        }:
            raise RuntimeError(f"invalid multiclass member specification at index {index}")
        filename = member["file"]
        iterations = member["iterations"]
        weight = member["weight"]
        expected_sha256 = member["sha256"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".txt")
        ):
            raise RuntimeError(f"invalid multiclass model filename at index {index}")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise RuntimeError(f"invalid multiclass iteration count at index {index}")
        if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            raise RuntimeError(f"invalid multiclass member weight at index {index}")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise RuntimeError(f"invalid multiclass model hash at index {index}")
        model_path = HERE / "model_multi" / filename
        if not model_path.is_file():
            raise RuntimeError(f"multiclass model file is missing: {filename}")
        actual_sha256 = sha256_path(model_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"multiclass model hash mismatch for {filename}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        booster = lgb.Booster(model_file=str(model_path))
        prediction = np.asarray(
            booster.predict(matrix, num_iteration=iterations), dtype=np.float64
        )
        if prediction.ndim != 2 or prediction.shape != (len(test), 4):
            raise RuntimeError(
                f"multiclass member {index} expected {(len(test), 4)}, "
                f"got {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"multiclass member {index} produced non-finite output")
        member_predictions.append(prediction[:, 3])
        member_weights.append(float(weight))
    weight_sum = float(sum(member_weights))
    if not np.isclose(weight_sum, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"multiclass member weights must sum to 1, got {weight_sum}")
    raw = np.average(
        np.column_stack(member_predictions), axis=1, weights=member_weights
    )
    if not np.isfinite(raw).all():
        raise RuntimeError("multiclass ensemble produced non-finite output")

    game_type = test["game_type"].astype(str).to_numpy()
    logits = np.log(np.clip(raw, EPS, 1 - EPS) / np.clip(1 - raw, EPS, 1 - EPS))
    multi_shifts = config.get("multi_shifts")
    if not isinstance(multi_shifts, dict) or set(multi_shifts) != {"R", "F"}:
        raise RuntimeError("multiclass shifts must contain exactly R and F")
    for group, delta in multi_shifts.items():
        if not isinstance(delta, (int, float)) or not np.isfinite(delta):
            raise RuntimeError(f"invalid multiclass shift for {group}")
        mask = game_type == group
        logits[mask] += float(delta)
    calibrated = 1.0 / (1.0 + np.exp(-logits))

    row_prior = prod._row_prior(test, meta)
    prediction = base + float(config["multi_coefficient"]) * (calibrated - row_prior)
    prediction = np.clip(prediction, EPS, 1.0 - EPS)
    if not np.isfinite(prediction).all():
        raise RuntimeError("blend produced non-finite probabilities")

    neural = config.get("neural_member")
    if neural is not None:
        required = {"files", "alpha", "mode", "status"}
        if not isinstance(neural, dict) or not required.issubset(neural):
            raise RuntimeError(f"incomplete neural member config: {neural!r}")
        if neural["mode"] != "logit_deviation":
            raise RuntimeError(f"unsupported neural member mode: {neural['mode']!r}")
        alpha = neural["alpha"]
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise RuntimeError("neural member alpha must be numeric")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 0.5:
            raise RuntimeError(f"neural member alpha out of range: {alpha}")
        alpha_by_group = neural.get("alpha_by_group")
        if alpha_by_group is not None:
            if not isinstance(alpha_by_group, dict) or set(alpha_by_group) != {"R", "F"}:
                raise RuntimeError("neural alpha_by_group must contain exactly R and F")
            for group, value in alpha_by_group.items():
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                    or not 0.0 <= float(value) <= 0.5
                ):
                    raise RuntimeError(f"neural alpha_by_group[{group}] out of range: {value}")
        specifications = neural["files"]
        if not isinstance(specifications, list) or not specifications:
            raise RuntimeError("neural member declares no files")
        if alpha > 0.0:
            # Grouped-4 neural inference.  All 64 original members remain and
            # are averaged in their original blend order.  Shared preprocessing
            # is used only after exact runtime contract verification.
            bundles = []
            reference_bundle = None
            reference_filename = None

            def _same_preprocessing(left, right):
                if left["names"] != right["names"]:
                    return False
                if left["dense_columns"] != right["dense_columns"]:
                    return False
                if left["maps"] != right["maps"]:
                    return False
                if not np.array_equal(
                    np.asarray(left["mean"]), np.asarray(right["mean"]), equal_nan=True
                ):
                    return False
                if not np.array_equal(
                    np.asarray(left["std"]), np.asarray(right["std"]), equal_nan=True
                ):
                    return False
                return True

            for index, specification in enumerate(specifications):
                if set(specification) != {"file", "sha256"}:
                    raise RuntimeError(f"invalid neural member spec at index {index}")
                filename = specification["file"]
                if (
                    not isinstance(filename, str)
                    or Path(filename).name != filename
                    or not filename.endswith(".pkl")
                ):
                    raise RuntimeError(f"invalid neural model filename at index {index}")
                model_path = HERE / "model_multi" / filename
                if not model_path.is_file():
                    raise RuntimeError(f"neural model file is missing: {filename}")

                payload = model_path.read_bytes()
                actual_sha256 = hashlib.sha256(payload).hexdigest()
                if actual_sha256 != specification["sha256"]:
                    raise RuntimeError(
                        f"neural model hash mismatch for {filename}: "
                        f"expected={specification['sha256']} actual={actual_sha256}"
                    )
                bundle = pickle.loads(payload)
                del payload

                if reference_bundle is None:
                    reference_bundle = bundle
                    reference_filename = filename
                elif not _same_preprocessing(reference_bundle, bundle):
                    raise RuntimeError(
                        "neural preprocessing differs across final-fit seeds: "
                        f"reference={reference_filename} member={filename}; "
                        "grouped inference is unsafe"
                    )
                bundles.append({"names": bundle["names"], "state": bundle["state"]})

            if reference_bundle is None or len(bundles) != len(specifications):
                raise RuntimeError("grouped neural path failed to load every member")

            def _stack_group(member_group):
                states = [member["state"] for member in member_group]
                names = member_group[0]["names"]
                grouped = {
                    "names": names,
                    "size": len(states),
                    "embeddings": [
                        np.stack(
                            [state[f"embeddings.{i}.weight"] for state in states],
                            axis=0,
                        )
                        for i in range(len(names))
                    ],
                }
                layer_index = 0
                layers = []
                while f"trunk.{layer_index}.weight" in states[0]:
                    keys = (
                        f"trunk.{layer_index}.weight",
                        f"trunk.{layer_index}.bias",
                        f"trunk.{layer_index + 1}.weight",
                        f"trunk.{layer_index + 1}.bias",
                        f"trunk.{layer_index + 1}.running_mean",
                        f"trunk.{layer_index + 1}.running_var",
                    )
                    for state in states[1:]:
                        if any(key not in state for key in keys):
                            raise RuntimeError(
                                f"neural architecture mismatch at trunk layer {layer_index}"
                            )
                    layers.append({
                        "weight": np.stack([state[keys[0]] for state in states], axis=0),
                        "bias": np.stack([state[keys[1]] for state in states], axis=0),
                        "gamma": np.stack([state[keys[2]] for state in states], axis=0),
                        "beta": np.stack([state[keys[3]] for state in states], axis=0),
                        "mean": np.stack([state[keys[4]] for state in states], axis=0),
                        "var": np.stack([state[keys[5]] for state in states], axis=0),
                    })
                    layer_index += 4
                grouped["layers"] = layers
                grouped["head_weight"] = np.stack(
                    [state["head.weight"] for state in states], axis=0
                )
                grouped["head_bias"] = np.stack(
                    [state["head.bias"] for state in states], axis=0
                )
                return grouped

            def _grouped_forward(group, codes, dense):
                embedding_pieces = [
                    table[:, codes[:, i], :]
                    for i, table in enumerate(group["embeddings"])
                ]
                embedded = np.concatenate(embedding_pieces, axis=2).astype(
                    np.float32, copy=False
                )
                first = group["layers"][0]
                embedding_width = embedded.shape[2]
                weight = first["weight"]
                weight_embedding = weight[:, :, :embedding_width]
                weight_dense = weight[:, :, embedding_width:]
                values = np.matmul(
                    embedded, np.swapaxes(weight_embedding, 1, 2)
                )
                values += np.matmul(
                    dense[None, :, :], np.swapaxes(weight_dense, 1, 2)
                )
                values += first["bias"][:, None, :]
                values = (
                    values - first["mean"][:, None, :]
                ) / np.sqrt(first["var"][:, None, :] + 1e-5)
                values = (
                    values * first["gamma"][:, None, :]
                    + first["beta"][:, None, :]
                )
                values = values / (1.0 + np.exp(-values))

                for layer in group["layers"][1:]:
                    values = (
                        np.matmul(values, np.swapaxes(layer["weight"], 1, 2))
                        + layer["bias"][:, None, :]
                    )
                    values = (
                        values - layer["mean"][:, None, :]
                    ) / np.sqrt(layer["var"][:, None, :] + 1e-5)
                    values = (
                        values * layer["gamma"][:, None, :]
                        + layer["beta"][:, None, :]
                    )
                    values = values / (1.0 + np.exp(-values))

                logits_grouped = (
                    np.matmul(
                        values,
                        np.swapaxes(group["head_weight"], 1, 2),
                    )
                    + group["head_bias"][:, None, :]
                )
                logits_grouped = logits_grouped - logits_grouped.max(
                    axis=2, keepdims=True
                )
                probabilities = np.exp(logits_grouped)
                probabilities /= probabilities.sum(axis=2, keepdims=True)
                return probabilities[:, :, 3]

            neural_group_size = 2
            groups = [
                _stack_group(bundles[start:start + neural_group_size])
                for start in range(0, len(bundles), neural_group_size)
            ]
            if sum(group["size"] for group in groups) != len(bundles):
                raise RuntimeError("grouped neural member count mismatch")

            pooled_blocks = []
            for start in range(0, len(test), NEURAL_BLOCK):
                stop = min(start + NEURAL_BLOCK, len(test))
                span = stop - start
                frame_block = test.iloc[start:stop]
                matrix_block = matrix.iloc[start:stop]
                codes = multi.nn_codes(frame_block, matrix_block, reference_bundle)
                dense = multi.nn_dense(matrix_block, reference_bundle)
                # Preserve the original (rows x members) averaging layout.  Each
                # grouped kernel returns float32 probabilities just like the
                # original per-member NumPy forward path; convert each result to
                # float64 before the final np.mean exactly as production did.
                block_members = np.empty((span, len(bundles)), dtype=np.float64)
                cursor = 0
                for group in groups:
                    grouped = np.asarray(
                        _grouped_forward(group, codes, dense), dtype=np.float32
                    )
                    count = grouped.shape[0]
                    if grouped.shape != (count, span):
                        raise RuntimeError(
                            f"invalid grouped neural output shape: {grouped.shape}"
                        )
                    member_block = grouped[:, :span]
                    if not np.isfinite(member_block).all():
                        raise RuntimeError("grouped neural member produced non-finite output")
                    if (member_block <= 0.0).any() or (member_block >= 1.0).any():
                        raise RuntimeError("grouped neural member produced a non-probability")
                    block_members[:, cursor:cursor + count] = member_block.T.astype(
                        np.float64, copy=False
                    )
                    cursor += count
                if cursor != len(bundles):
                    raise RuntimeError(
                        f"grouped neural block member count mismatch: {cursor}"
                    )
                pooled_blocks.append(np.mean(block_members, axis=1))

            pooled = np.concatenate(pooled_blocks)
            if pooled.shape != (len(test),) or not np.isfinite(pooled).all():
                raise RuntimeError("grouped neural pooled output is invalid")
            pooled = np.clip(pooled, EPS, 1 - EPS)
            deviation = np.log(pooled / (1.0 - pooled))
            weight = np.full(len(test), alpha, dtype=np.float64)
            if alpha_by_group is not None:
                for group, value in alpha_by_group.items():
                    weight[game_type == group] = float(value)
            current = np.clip(prediction, EPS, 1 - EPS)
            shifted = np.log(current / (1.0 - current)) + weight * deviation
            prediction = 1.0 / (1.0 + np.exp(-shifted))
            prediction = np.clip(prediction, EPS, 1.0 - EPS)
            if not np.isfinite(prediction).all():
                raise RuntimeError("neural blend produced non-finite probabilities")

    # Count-segment specialist.  The champion fits one global tree ensemble with
    # the count as a feature, so split capacity goes where the global loss falls
    # fastest and the two-strike regime -- 28.8% of rows, the worst-resolved
    # segment in the Murphy decomposition -- is starved.  This blends in a
    # LightGBM trained only on that segment, on the same shipped 331-column decay
    # contract, at a weight selected on seasons 2022 and 2024 only.
    #
    # It enters as a convex combination rather than an added member, so the
    # blended logit spread stays between the two inputs instead of growing; that
    # is why it needs no rescaling constant.  Each row uses only its own
    # strikes_before and its own feature row.
    specialist_path = HERE / "strike_specialist.json"
    if specialist_path.is_file():
        specialist_cfg = json.loads(specialist_path.read_text(encoding="utf-8"))
        if specialist_cfg.get("schema_version") != 1:
            raise RuntimeError(
                f"unsupported strike_specialist schema: {specialist_cfg.get('schema_version')}")
        if specialist_cfg.get("space") != "logit":
            raise RuntimeError("strike specialist must blend in logit space")
        if specialist_cfg.get("column") != "strikes_before":
            raise RuntimeError("strike specialist must key on strikes_before")
        if bool(specialist_cfg.get("uses_test_rows", True)):
            raise RuntimeError("strike specialist must not read evaluation rows")
        segment_value = specialist_cfg.get("strikes_before")
        if not isinstance(segment_value, int) or not 0 <= segment_value <= 2:
            raise RuntimeError(f"invalid strike specialist segment: {segment_value!r}")
        blend_weight = specialist_cfg.get("weight")
        if (
            not isinstance(blend_weight, (int, float))
            or isinstance(blend_weight, bool)
            or not np.isfinite(blend_weight)
            or not 0.0 <= float(blend_weight) <= 0.5
        ):
            raise RuntimeError(f"strike specialist weight out of range: {blend_weight!r}")
        blend_weight = float(blend_weight)
        specifications = specialist_cfg.get("files")
        if not isinstance(specifications, list) or not specifications:
            raise RuntimeError("strike specialist declares no models")
        if blend_weight > 0.0:
            decay_matrix = captured.get("lgb_decay_matrix")
            if decay_matrix is None:
                raise RuntimeError("the decay matrix was not captured from _prod.predict")
            if decay_matrix.shape[0] != len(test):
                raise RuntimeError("captured decay matrix does not match the test frame")
            strikes = test["strikes_before"].to_numpy()
            if not np.isfinite(strikes.astype(np.float64)).all():
                raise RuntimeError("strikes_before holds a non-finite value")
            segment_mask = strikes.astype(np.int64) == segment_value
            excluded = specialist_cfg.get("exclude_balls") or []
            if not isinstance(excluded, list):
                raise RuntimeError("exclude_balls must be a list")
            if excluded:
                balls = test["balls_before"].to_numpy()
                if not np.isfinite(balls.astype(np.float64)).all():
                    raise RuntimeError("balls_before holds a non-finite value")
                balls = balls.astype(np.int64)
                for value in excluded:
                    if not isinstance(value, int) or isinstance(value, bool):
                        raise RuntimeError(f"invalid excluded ball count: {value!r}")
                    segment_mask &= balls != value
            if segment_mask.any():
                segment_rows = decay_matrix[segment_mask]
                member_predictions = []
                for index, specification in enumerate(specifications):
                    if set(specification) != {"file", "sha256"}:
                        raise RuntimeError(f"invalid specialist spec at index {index}")
                    filename = specification["file"]
                    if (
                        not isinstance(filename, str)
                        or Path(filename).name != filename
                        or not filename.endswith(".txt")
                    ):
                        raise RuntimeError(f"invalid specialist filename at index {index}")
                    model_path = HERE / "model" / filename
                    if not model_path.is_file():
                        raise RuntimeError(f"specialist model is missing: {filename}")
                    actual = sha256_path(model_path)
                    if actual != specification["sha256"]:
                        raise RuntimeError(
                            f"specialist model hash mismatch for {filename}: "
                            f"expected={specification['sha256']} actual={actual}")
                    booster = lgb.Booster(model_file=str(model_path))
                    if booster.num_feature() != decay_matrix.shape[1]:
                        raise RuntimeError(
                            f"specialist expects {booster.num_feature()} features, "
                            f"the decay matrix has {decay_matrix.shape[1]}")
                    member = np.asarray(booster.predict(segment_rows), dtype=np.float64)
                    if member.shape != (int(segment_mask.sum()),) or not np.isfinite(member).all():
                        raise RuntimeError(f"specialist member {index} produced invalid output")
                    member_predictions.append(member)
                    del booster
                pooled = np.clip(np.mean(np.column_stack(member_predictions), axis=1),
                                 EPS, 1.0 - EPS)
                current = np.clip(prediction[segment_mask], EPS, 1.0 - EPS)
                blended = (
                    (1.0 - blend_weight) * np.log(current / (1.0 - current))
                    + blend_weight * np.log(pooled / (1.0 - pooled))
                )
                prediction[segment_mask] = np.clip(
                    1.0 / (1.0 + np.exp(-blended)), EPS, 1.0 - EPS)
                if not np.isfinite(prediction).all():
                    raise RuntimeError("strike specialist produced non-finite probabilities")

    # Existing evaluation-level shift remains after the newly validated count
    # residual, matching the current shipped 1144.5 lineage except for this one
    # added row-local correction.
    level_path = HERE / "level_shift.json"
    if level_path.is_file():
        level = json.loads(level_path.read_text(encoding="utf-8"))
        if level.get("schema_version") != 1:
            raise RuntimeError(
                f"unsupported level_shift schema: {level.get('schema_version')}"
            )
        if level.get("space") != "probability":
            raise RuntimeError(f"unsupported level_shift space: {level.get('space')!r}")
        shifts = level.get("shifts")
        if not isinstance(shifts, dict) or set(shifts) != {"R", "F"}:
            raise RuntimeError("level_shift shifts must contain exactly R and F")
        offset = np.zeros(len(prediction), dtype=np.float64)
        for group, value in shifts.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
            ):
                raise RuntimeError(f"invalid level shift for {group}")
            if abs(float(value)) > 0.05:
                raise RuntimeError(f"level shift for {group} out of range: {value}")
            offset[game_type == group] = float(value)
        if np.any(offset):
            prediction = np.clip(prediction + offset, EPS, 1.0 - EPS)
            if not np.isfinite(prediction).all():
                raise RuntimeError("level shift produced non-finite probabilities")

    indexed = pd.Series(prediction, index=test[ID_COL].to_numpy())
    ordered = indexed.reindex(submission[ID_COL]).to_numpy(np.float64)
    if not np.isfinite(ordered).all():
        raise RuntimeError("row_id alignment produced missing predictions")
    submission[TARGET_COL] = np.clip(ordered, EPS, 1.0 - EPS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

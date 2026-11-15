"""Inference-only follow-up analysis for the Guardian Vision test split.

This script loads existing checkpoints, never trains or writes checkpoint files,
and writes all analysis artifacts beside this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluate_test_v3 import (
    CONFIDENCE_THRESHOLD,
    FX,
    FY,
    IOU_THRESHOLD,
    GroundTruth,
    distance_from_prediction,
    iou,
    load_test_rows,
    parse_ground_truth,
    zone,
)

LOGGER = logging.getLogger(__name__)
IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375
BINS = ((0.0, 10.0, "0_10"), (10.0, 20.0, "10_20"), (20.0, 30.0, "20_30"), (30.0, float("inf"), "30_plus"))
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261109


def match(
    truths: list[GroundTruth],
    predictions: list[tuple[tuple[float, float, float, float], float]],
    clip_truth: bool = False,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return confidence-ordered one-to-one matches."""
    clipped = []
    for truth in truths:
        box = truth.bbox
        if clip_truth:
            box = (
                max(0.0, min(float(IMAGE_WIDTH), box[0])),
                max(0.0, min(float(IMAGE_HEIGHT), box[1])),
                max(0.0, min(float(IMAGE_WIDTH), box[2])),
                max(0.0, min(float(IMAGE_HEIGHT), box[3])),
            )
        clipped.append(GroundTruth(box, truth.lateral_m, truth.forward_m, truth.occlusion))
    pred_to_truth: dict[int, int] = {}
    truth_to_pred: dict[int, int] = {}
    for pred_index in sorted(range(len(predictions)), key=lambda index: -predictions[index][1]):
        choices = [
            (iou(predictions[pred_index][0], truth.bbox), truth_index)
            for truth_index, truth in enumerate(clipped)
            if truth_index not in truth_to_pred
        ]
        if choices:
            score, truth_index = max(choices)
            if score >= IOU_THRESHOLD:
                pred_to_truth[pred_index] = truth_index
                truth_to_pred[truth_index] = pred_index
    return pred_to_truth, truth_to_pred


def infer(
    model_path: Path,
    rows: list[tuple[str, Path, str]],
    labels_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Run fixed-threshold inference and retain boxes for the follow-up."""
    model = YOLO(str(model_path))
    output: dict[str, dict[str, Any]] = {}
    for index, (frame_id, image_path, sequence_id) in enumerate(rows, start=1):
        truths = parse_ground_truth(labels_dir / f"{frame_id}.txt")
        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=1280,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )[0]
        predictions = [
            (tuple(float(value) for value in box.xyxy[0].tolist()), float(box.conf[0]))
            for box in result.boxes
            if int(box.cls[0]) == 0
        ]
        output[frame_id] = {
            "frame_id": frame_id,
            "sequence_id": sequence_id,
            "image": str(image_path),
            "truths": truths,
            "predictions": predictions,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
        }
        if index % 100 == 0 or index == len(rows):
            LOGGER.info("%s: %d/%d", model_path.name, index, len(rows))
    return output


def error_stats(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Summarize signed prediction-minus-ground-truth errors."""
    if not pairs:
        return {"count": 0, "mae_m": float("nan"), "absrel": float("nan"), "bias_m": float("nan"), "median_error_m": float("nan")}
    errors = np.asarray([pred - truth for truth, pred in pairs], dtype=float)
    truths = np.asarray([truth for truth, _ in pairs], dtype=float)
    return {
        "count": int(len(errors)),
        "mae_m": float(np.mean(np.abs(errors))),
        "absrel": float(np.mean(np.abs(errors) / np.maximum(truths, 1e-8))),
        "bias_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
    }


def bootstrap(
    sequence_values: dict[str, list[float]],
    statistic: str = "mean",
) -> list[float]:
    """Return sequence-bootstrap replicates for scalar values."""
    if not sequence_values:
        return []
    groups = list(sequence_values.values())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        flattened = np.asarray([value for group in sample for value in group], dtype=float)
        if not len(flattened):
            continue
        values.append(float(np.mean(flattened)) if statistic == "mean" else float(np.mean(flattened)))
    return values


def ci(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else [float("nan"), float("nan")]


def paired_distance(
    pretrained: dict[str, dict[str, Any]],
    finetuned: dict[str, dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare distances only where both models match the same GT object."""
    overall: dict[str, list[tuple[float, float]]] = {"pretrained": [], "finetuned": []}
    by_bin: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: {"pretrained": [], "finetuned": []})
    by_height: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_truncation: dict[str, list[tuple[float, float]]] = defaultdict(list)
    common_count = 0
    for frame_id, first in pretrained.items():
        second = finetuned[frame_id]
        first_match, _ = match(first["truths"], first["predictions"])
        second_match, _ = match(second["truths"], second["predictions"])
        common_truths = set(first_match.values()) & set(second_match.values())
        common_count += len(common_truths)
        for truth_index in common_truths:
            truth = first["truths"][truth_index]
            first_pred = first["predictions"][next(index for index, value in first_match.items() if value == truth_index)][0]
            second_pred = second["predictions"][next(index for index, value in second_match.items() if value == truth_index)][0]
            first_distance = distance_from_prediction(first_pred)
            second_distance = distance_from_prediction(second_pred)
            if first_distance is None or second_distance is None:
                continue
            overall["pretrained"].append((truth.forward_m, first_distance))
            overall["finetuned"].append((truth.forward_m, second_distance))
            bin_name = next(name for low, high, name in BINS if low <= truth.forward_m < high)
            by_bin[bin_name]["pretrained"].append((truth.forward_m, first_distance))
            by_bin[bin_name]["finetuned"].append((truth.forward_m, second_distance))
            height = truth.bbox[3] - truth.bbox[1]
            height_bin = "0_30px" if height < 30 else "30_60px" if height < 60 else "60_120px" if height < 120 else "120_plus"
            by_height[height_bin].append((truth.forward_m, second_distance))
            truncation_value = raw[frame_id][truth_index]["truncation"]
            truncation = "0" if truncation_value == 0 else "0_to_0.25" if truncation_value <= 0.25 else "0.25_to_0.5" if truncation_value <= 0.5 else "over_0.5"
            by_truncation[truncation].append((truth.forward_m, second_distance))
    return {
        "common_matched_pedestrians": common_count,
        "overall": {name: error_stats(values) for name, values in overall.items()},
        "by_ground_truth_distance_bin": {
            name: {model: error_stats(values) for model, values in models.items()}
            for name, models in by_bin.items()
        },
        "finetuned_bias_by_box_height_px": {name: error_stats(values) for name, values in by_height.items()},
        "finetuned_bias_by_truncation": {name: error_stats(values) for name, values in by_truncation.items()},
    }


def load_raw_labels(labels_dir: Path, frame_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Read raw KITTI fields needed for truncation, height, and diagnostics."""
    parsed: dict[str, list[dict[str, Any]]] = {}
    for frame_id in frame_ids:
        rows = []
        path = labels_dir / f"{frame_id}.txt"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 15 and fields[0].lower() == "pedestrian":
                    rows.append({
                        "bbox": tuple(float(value) for value in fields[4:8]),
                        "truncation": float(fields[1]),
                        "occlusion": int(float(fields[2])),
                        "height_3d_m": float(fields[8]),
                        "forward_m": float(fields[13]),
                    })
        parsed[frame_id] = rows
    return parsed


def geometry_analysis(
    train_labels: Path,
    training_ids: set[str],
    raw_test: dict[str, list[dict[str, Any]]],
    finetuned: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fit height and affine corrections from training labels only."""
    factors = []
    silhouette_ratios = []
    real_heights = []
    for frame_id in training_ids:
        for row in load_raw_labels(train_labels, {frame_id})[frame_id]:
            box_height = row["bbox"][3] - row["bbox"][1]
            if box_height > 0 and row["forward_m"] > 0:
                factors.append(box_height * row["forward_m"] / FY)
                real_heights.append(row["height_3d_m"])
                expected_full_height = FY * row["height_3d_m"] / row["forward_m"]
                if expected_full_height > 0:
                    silhouette_ratios.append(box_height / expected_full_height)
    learned_height = float(np.median(factors))
    train_rows = [row for frame_id in training_ids for row in load_raw_labels(train_labels, {frame_id})[frame_id]]
    affine_x = np.asarray([FY * 1.7 / (row["bbox"][3] - row["bbox"][1]) for row in train_rows if row["bbox"][3] > row["bbox"][1]], dtype=float)
    affine_y = np.asarray([row["forward_m"] for row in train_rows if row["bbox"][3] > row["bbox"][1]], dtype=float)
    if len(affine_x):
        design = np.column_stack((affine_x, np.ones(len(affine_x))))
        affine_a, affine_b = np.linalg.lstsq(design, affine_y, rcond=None)[0]
    else:
        affine_a, affine_b = 1.0, 0.0
    baseline_pairs = []
    height_pairs = []
    affine_pairs = []
    for frame in finetuned.values():
        _, truth_to_pred = match(frame["truths"], frame["predictions"])
        for truth_index, pred_index in truth_to_pred.items():
            truth = frame["truths"][truth_index]
            prediction = frame["predictions"][pred_index][0]
            box_height = prediction[3] - prediction[1]
            if box_height <= 0:
                continue
            baseline = FY * 1.7 / box_height
            corrected = FY * learned_height / box_height
            affine = affine_a * baseline + affine_b
            baseline_pairs.append((truth.forward_m, baseline))
            height_pairs.append((truth.forward_m, corrected))
            affine_pairs.append((truth.forward_m, affine))
    return {
        "training_label_count": len(train_rows),
        "training_real_height_m": {"mean": float(np.mean(real_heights)), "median": float(np.median(real_heights)), "count": len(real_heights), "source": "KITTI/CARLA pedestrian height field"},
        "training_box_height_times_distance_over_fy_m": {"mean": float(np.mean(factors)), "median": learned_height, "count": len(factors)},
        "training_box_height_to_projected_silhouette_ratio": {"mean": float(np.mean(silhouette_ratios)), "median": float(np.median(silhouette_ratios)), "count": len(silhouette_ratios)},
        "learned_height_m": learned_height,
        "test_affine_correction": {"a": float(affine_a), "b_m": float(affine_b), "fit_source": "TRAIN labels only (reported for transparency; coefficients evaluated on TEST matched predictions)"},
        "test_geometric_re_evaluation_finetuned": {
            "fixed_1.7m": error_stats(baseline_pairs),
            "learned_height": error_stats(height_pairs),
            "affine": error_stats(affine_pairs),
        },
    }


def denominator_report(frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Count critical denominators and false-brake numerators."""
    report = {}
    for name, corridor in (("all_pedestrians", False), ("corridor_2.0m", True)):
        sequence_rows: dict[str, list[dict[str, int]]] = defaultdict(list)
        critical_total = critical_detected = predicted_critical = false_objects = false_frames = 0
        frame_total = 0
        for frame in frames.values():
            truths = [truth for truth in frame["truths"] if not corridor or abs(truth.lateral_m) <= 2.0 and truth.forward_m <= 30.0]
            predictions = [pred for pred in frame["predictions"] if not corridor or _prediction_in_corridor(pred[0])]
            pred_to_truth, truth_to_pred = match(truths, predictions)
            critical = [index for index, truth in enumerate(truths) if zone(truth.forward_m) == "critical_danger"]
            frame_critical_total = len(critical)
            frame_critical_detected = sum(
                index in truth_to_pred and zone(distance_from_prediction(predictions[truth_to_pred[index]][0]) or float("inf")) == "critical_danger"
                for index in critical
            )
            frame_predicted_critical = sum(
                zone(distance_from_prediction(prediction[0]) or float("inf")) == "critical_danger"
                for prediction in predictions
            )
            critical_pred_indices = [index for index, prediction in enumerate(predictions) if zone(distance_from_prediction(prediction[0]) or float("inf")) == "critical_danger"]
            frame_false_objects = sum(index not in pred_to_truth or zone(truths[pred_to_truth[index]].forward_m) != "critical_danger" for index in critical_pred_indices)
            frame_false_brake = int(bool(critical_pred_indices) and not bool(critical))
            critical_total += frame_critical_total
            critical_detected += frame_critical_detected
            predicted_critical += frame_predicted_critical
            false_objects += frame_false_objects
            false_frames += frame_false_brake
            frame_total += 1
            sequence_rows[frame["sequence_id"]].append({
                "critical_detected": frame_critical_detected,
                "critical_total": frame_critical_total,
                "false_objects": frame_false_objects,
                "predicted_critical": frame_predicted_critical,
                "false_frames": frame_false_brake,
                "frames": 1,
            })
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        recalls = []
        object_rates = []
        frame_rates = []
        groups = list(sequence_rows.values())
        for _ in range(BOOTSTRAP_REPLICATES):
            sample = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
            flat = [row for group in sample for row in group]
            detected = sum(row["critical_detected"] for row in flat)
            total = sum(row["critical_total"] for row in flat)
            false_objects_sample = sum(row["false_objects"] for row in flat)
            predicted_sample = sum(row["predicted_critical"] for row in flat)
            false_frames_sample = sum(row["false_frames"] for row in flat)
            frames_sample = sum(row["frames"] for row in flat)
            recalls.append(detected / total if total else 0.0)
            object_rates.append(false_objects_sample / predicted_sample if predicted_sample else 0.0)
            frame_rates.append(false_frames_sample / frames_sample if frames_sample else 0.0)
        report[name] = {
            "critical_recall": {"detected": critical_detected, "denominator": critical_total, "value": critical_detected / critical_total if critical_total else None},
            "false_brake_per_object": {"false_brake_objects": false_objects, "predicted_critical_objects": predicted_critical, "value": false_objects / predicted_critical if predicted_critical else None},
            "false_brake_per_frame": {"false_brake_frames": false_frames, "frames": frame_total, "value": false_frames / frame_total if frame_total else None},
            "bootstrap_95_ci": {
                "critical_recall": ci(recalls),
                "false_brake_per_object": ci(object_rates),
                "false_brake_per_frame": ci(frame_rates),
            },
        }
    return report


def _prediction_in_corridor(prediction: tuple[float, float, float, float]) -> bool:
    distance = distance_from_prediction(prediction)
    if distance is None:
        return False
    center_x = (prediction[0] + prediction[2]) / 2.0
    return abs((center_x - 621.0) * distance / FX) <= 2.0 and distance <= 30.0


def diagnostic_report(
    frames: dict[str, dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Report 0-10 m misses and create annotated worst-miss images."""
    counts = defaultdict(int)
    worst: list[tuple[float, str, int]] = []
    clipped_counts = {"matched": 0, "unmatched": 0}
    for frame_id, frame in frames.items():
        truths = frame["truths"]
        pred_to_truth, truth_to_pred = match(truths, frame["predictions"])
        _, clipped_truth_to_pred = match(truths, frame["predictions"], clip_truth=True)
        for index, truth in enumerate(truths):
            if not 0 <= truth.forward_m < 10:
                continue
            if index in truth_to_pred:
                continue
            row = raw[frame_id][index] if index < len(raw[frame_id]) else {"truncation": None, "occlusion": None, "bbox": truth.bbox}
            counts["misses"] += 1
            counts[f"truncation_{row['truncation']}"] += 1
            counts[f"occlusion_{row['occlusion']}"] += 1
            if truth.bbox[0] < 0 or truth.bbox[1] < 0 or truth.bbox[2] > IMAGE_WIDTH or truth.bbox[3] > IMAGE_HEIGHT:
                counts["extends_past_image_border"] += 1
            overlaps = max((iou(truth.bbox, other.bbox) for other_index, other in enumerate(truths) if other_index != index), default=0.0)
            counts["overlap_with_other_pedestrian"] += int(overlaps > 0.1)
            worst.append((overlaps, frame_id, index))
        clipped_counts["matched"] += len(clipped_truth_to_pred)
        clipped_counts["unmatched"] += len(truths) - len(clipped_truth_to_pred)
    worst.sort(reverse=True)
    image_dir = output_dir / "worst_misses"
    image_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, frame_id, truth_index) in enumerate(worst[:30], start=1):
        frame = frames[frame_id]
        image = Image.open(frame["image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for truth in frame["truths"]:
            draw.rectangle(truth.bbox, outline="red", width=2)
        for prediction, _ in frame["predictions"]:
            draw.rectangle(prediction, outline="green", width=2)
        image.save(image_dir / f"{rank:02d}_{frame_id}_miss_{truth_index}.png")
    return {"zero_to_ten_misses": dict(counts), "clip_to_image_matching": clipped_counts, "saved_worst_misses": min(30, len(worst)), "note": "Red boxes are ground truth; green boxes are predictions from the fine-tuned model."}


def bin_protocol_report(frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Report GT-bin matched pairs and predicted-bin unmatched predictions separately."""
    matched = defaultdict(lambda: {"tp": 0, "fn": 0})
    unmatched_predictions = defaultdict(int)
    for frame in frames.values():
        pred_to_truth, truth_to_pred = match(frame["truths"], frame["predictions"])
        for index, truth in enumerate(frame["truths"]):
            name = next(name for low, high, name in BINS if low <= truth.forward_m < high)
            matched[name]["tp" if index in truth_to_pred else "fn"] += 1
        for prediction_index, prediction in enumerate(frame["predictions"]):
            if prediction_index not in pred_to_truth:
                distance = distance_from_prediction(prediction[0])
                if distance is not None:
                    name = next(name for low, high, name in BINS if low <= distance < high)
                    unmatched_predictions[name] += 1
    return {"ground_truth_bin_matched_pairs": dict(matched), "unmatched_predictions_by_predicted_distance_bin": dict(unmatched_predictions), "protocol": "Each matched pair is assigned once using ground-truth distance; unmatched predictions are counted separately using predicted geometric distance."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=Path("runs/detect/outputs/training/guardian_yolo11n-3/weights/best.pt"))
    parser.add_argument("--pretrained", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/yolo/test.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata.csv"))
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results_v3/followup"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_test_rows(args.manifest, args.metadata, args.images)
    if len(rows) != 994:
        raise ValueError(f"Expected 994 test frames, found {len(rows)}")
    pretrained = infer(args.pretrained, rows, args.labels)
    finetuned = infer(args.best, rows, args.labels)
    raw_test = load_raw_labels(args.labels, set(finetuned))
    training_ids = {
        f"{int(row['frame_id']):06d}"
        for row in csv.DictReader(Path("dataset/metadata.csv").open(newline="", encoding="utf-8"))
        if row["split"] == "train"
    }
    training_dir = args.labels
    paired = paired_distance(pretrained, finetuned, raw_test)
    geometry = geometry_analysis(training_dir, training_ids, raw_test, finetuned)
    diagnostics = diagnostic_report(finetuned, raw_test, args.output)
    protocol = bin_protocol_report(finetuned)
    denominators = {"pretrained": denominator_report(pretrained), "finetuned_best": denominator_report(finetuned)}
    provenance = {
        "active_checkpoint": str(args.best),
        "active_checkpoint_run": "guardian_yolo11n-3",
        "active_checkpoint_args": str(Path("runs/detect/outputs/training/guardian_yolo11n-3/args.yaml")),
        "pretrained_checkpoint": str(args.pretrained),
        "data_yaml": str(Path("dataset/yolo/carla.yaml")),
        "split": "dataset/yolo/test.txt; 994 frames; sequence-disjoint test split",
        "fixed_inference": {"imgsz": 1280, "confidence": CONFIDENCE_THRESHOLD, "iou": IOU_THRESHOLD},
        "older_numbers": {"map50": 0.9196, "map50_95": 0.6195, "status": "SUPERSEDED pending provenance; do not compare as current test result unless its original split/checkpoint is reproduced"},
        "current_numbers": {"map50": 0.85298, "map50_95": 0.58323, "source": "outputs/results_v3/ultralytics_val_metrics.json"},
        "training_frame_ids_used_for_geometry_fit": len(training_ids),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    shutil.copy2(Path("runs/detect/outputs/training/guardian_yolo11n-3/args.yaml"), args.output / "guardian_yolo11n-3_args.yaml")
    (args.output / "paired_distance.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (args.output / "geometry_bias.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    (args.output / "critical_denominators.json").write_text(json.dumps(denominators, indent=2), encoding="utf-8")
    (args.output / "zero_to_ten_diagnosis.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (args.output / "distance_bin_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (args.output / "BUG_FIXES.txt").write_text(
        "Follow-up bug fixes and protocol corrections\n"
        "1. Provenance is recorded for the active YAML, test manifest, and guardian_yolo11n-3 checkpoint.\n"
        "2. Paired distance metrics use only GT pedestrians matched independently by both models.\n"
        "3. Distance-bin matched pairs are assigned by ground-truth distance exactly once; unmatched predictions are reported separately by predicted distance.\n"
        "4. Critical-zone denominators and numerators are printed for all pedestrians and the 2.0 m corridor, with sequence-bootstrap confidence intervals available from the source reports.\n"
        "5. A clipped-ground-truth matching pass and 30 annotated worst misses were added without changing source labels or checkpoints.\n"
        "6. Geometry calibration is fit from training labels only and evaluated on the test split.\n",
        encoding="utf-8",
    )
    LOGGER.info("Follow-up artifacts written to %s", args.output)


if __name__ == "__main__":
    main()

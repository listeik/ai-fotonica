"""Apply the frozen carrier-study promotion rule to predeclared finalists."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

MEAN_GUARDS = ("mean_pcc", "mean_ssim", "mean_local_ssim")
TAIL_GUARDS = ("mean_worst_pcc", "p10_worst_joint")
NEW_COHORTS = ("fresh_broad_test", "external_natural_test")
EXPECTED_GROUP_WEIGHTS = {
    "fresh_broad_test": 0.5,
    "external_natural_test": 0.3,
    "people_test": 0.15,
    "mixed_test": 0.05,
}
EXPECTED_METRIC_WEIGHTS = {
    "mean_pcc": 0.35, "mean_global_ssim": 0.2, "mean_windowed_ssim": 0.2,
    "mean_worst_pcc": 0.15, "mean_worst_global_ssim": 0.1,
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quartet_values(row):
    planes = row["planes"]
    if len(planes) != 4:
        raise ValueError("Expected four planes per quartet")
    for plane in planes:
        if any(not math.isfinite(plane[key]) for key in ("pcc", "ssim", "local_ssim")):
            raise ValueError("Nonfinite quality metric")
    return {
        "mean_pcc": fmean(plane["pcc"] for plane in planes),
        "mean_ssim": fmean(plane["ssim"] for plane in planes),
        "mean_local_ssim": fmean(plane["local_ssim"] for plane in planes),
        "mean_worst_pcc": min(plane["pcc"] for plane in planes),
        "mean_worst_ssim": min(plane["ssim"] for plane in planes),
        "weakest_joint": min(min(plane["pcc"], plane["ssim"]) for plane in planes),
    }


def quartet_score(values):
    return (0.35 * values["mean_pcc"] + 0.2 * values["mean_ssim"]
            + 0.2 * values["mean_local_ssim"] + 0.15 * values["mean_worst_pcc"]
            + 0.1 * values["mean_worst_ssim"])


def summarize_quartets(rows):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Each primary group needs nonempty, uniquely identified quartets")
    values = [quartet_values(row) for row in rows]
    summary = {key: fmean(row[key] for row in values) for key in MEAN_GUARDS + TAIL_GUARDS[:1]}
    summary["p10_worst_joint"] = float(np.quantile([row["weakest_joint"] for row in values], 0.1))
    summary["group_score"] = fmean(quartet_score(row) for row in values)
    summary["quartets"] = len(rows)
    return summary, np.array([quartet_score(row) for row in values])


def check_pair(current, reference):
    if [row["id"] for row in current] != [row["id"] for row in reference]:
        raise ValueError("Candidate and reference quartet identifiers/order differ")
    for lhs, rhs in zip(current, reference, strict=True):
        for key in ("target_pixels_sha256", "plane_target_sha256"):
            if lhs.get(key) != rhs.get(key):
                raise ValueError("Candidate and reference target checksums differ")


def judge(comparison, protocol, names):
    weights = protocol["final_test"]["weights"]
    if weights != EXPECTED_GROUP_WEIGHTS or protocol["group_score"] != EXPECTED_METRIC_WEIGHTS:
        raise ValueError("Protocol weights differ from the frozen scoring implementation")
    baseline_name = comparison["baseline"]
    baseline = comparison["models"][baseline_name]
    if baseline["sha256"] != protocol["baseline_sha256"]:
        raise ValueError("Comparison baseline differs from the frozen production checkpoint")
    if comparison["test"]["sha256"] != protocol["test_sha256"]:
        raise ValueError("Comparison test cache differs from the frozen protocol")
    if not 1 <= len(names) <= 2 or len(set(names)) != len(names) or baseline_name in names:
        raise ValueError("Provide one or two distinct predeclared finalists, excluding the baseline")
    for name in names:
        if name not in comparison["models"]:
            raise ValueError(f"Finalist {name} is absent from the comparison")

    baseline_summaries = {}
    baseline_scores = {}
    for group in weights:
        rows = baseline["splits"][group]["quartets"]
        baseline_summaries[group], baseline_scores[group] = summarize_quartets(rows)
    baseline_weighted = sum(weights[group] * row["group_score"] for group, row in baseline_summaries.items())
    normalized = {group: weights[group] / sum(weights[k] for k in NEW_COHORTS) for group in NEW_COHORTS}
    resampling = {
        group: np.random.default_rng(20261125 + index).integers(
            len(baseline_scores[group]), size=(4000, len(baseline_scores[group]))
        )
        for index, group in enumerate(NEW_COHORTS)
    }
    results = {}
    for name in names:
        model = comparison["models"][name]
        summaries, scores, guards = {}, {}, {}
        reasons = []
        for group in weights:
            rows = model["splits"][group]["quartets"]
            check_pair(rows, baseline["splits"][group]["quartets"])
            summaries[group], scores[group] = summarize_quartets(rows)
            guards[group] = {}
            for metric in MEAN_GUARDS + TAIL_GUARDS:
                delta = summaries[group][metric] - baseline_summaries[group][metric]
                limit = -0.003 if metric in MEAN_GUARDS else -0.005
                passed = delta >= limit
                guards[group][metric] = {"delta": delta, "minimum_allowed_delta": limit, "pass": passed}
                if not passed:
                    reasons.append(f"{group}/{metric} delta {delta:.8f} is below {limit}")
        weighted = sum(weights[group] * row["group_score"] for group, row in summaries.items())
        gain = weighted - baseline_weighted
        if gain < 0.002:
            reasons.append(f"Weighted gain {gain:.8f} is below 0.002")
        draws = sum(
            normalized[group] * (scores[group] - baseline_scores[group])[resampling[group]].mean(1)
            for group in NEW_COHORTS
        )
        point = sum(normalized[group] * (scores[group] - baseline_scores[group]).mean() for group in NEW_COHORTS)
        low, high = np.quantile(draws, (0.025, 0.975))
        if low <= 0:
            reasons.append(f"New-cohort paired-bootstrap lower bound {low:.8f} is not positive")
        results[name] = {
            "eligible": not reasons, "reasons": reasons, "checkpoint_sha256": model["sha256"],
            "weighted_score": weighted, "weighted_gain": gain, "groups": summaries,
            "guard_deltas": guards,
            "new_cohort_composite": {"mean_delta": float(point), "paired_bootstrap_95": [float(low), float(high)]},
        }
    return {
        "baseline": baseline_name, "baseline_sha256": baseline["sha256"],
        "baseline_weighted_score": baseline_weighted, "baseline_groups": baseline_summaries,
        "predeclared_finalists": names, "finalists": results,
        "rule": {
            "group_weights": weights, "group_score": EXPECTED_METRIC_WEIGHTS,
            "minimum_gain": 0.002, "mean_metric_max_drop": 0.003, "tail_metric_max_drop": 0.005,
            "new_cohort_bootstrap": {"groups": normalized, "resamples": 4000, "seed": 20261125,
                                     "resampling": "separate cohort RNG streams; paired quartets stay intact",
                                     "required_lower_bound": ">0"},
            "external_quartet": "display-only; excluded from all promotion quantities",
            "action": "eligibility report only; no deployment or post-test model selection",
        },
        "limitations": [
            "Quartet bootstrap retains paired planes but cannot establish generalization to independent face identities.",
            "The ten portrait quartets reuse four people; historical groups share sources and were previously inspected.",
            "New broad and external natural cohorts are bootstrapped separately and are the only groups used for the interval.",
            "At most two predeclared finalists are assessed; intervals are unadjusted for multiple finalist comparisons.",
            "Eligibility is an operational frozen rule and does not establish a universal quality improvement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/carrier-study-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalist", action="append", required=True,
                        help="one or two names frozen before test evaluation; other models are ignored")
    args = parser.parse_args()
    if args.output.resolve() in {args.comparison.resolve(), args.protocol.resolve()}:
        parser.error("output must differ from comparison and protocol")
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = judge(comparison, protocol, args.finalist)
    report["inputs"] = {"comparison_sha256": file_hash(args.comparison), "protocol_sha256": file_hash(args.protocol),
                        "summarizer_sha256": file_hash(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: row[key] for key in ("eligible", "weighted_score", "weighted_gain", "reasons")}
                      for name, row in report["finalists"].items()}, allow_nan=False))


if __name__ == "__main__":
    main()

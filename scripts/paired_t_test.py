"""Compare NDCG@10 per user from two sets of shared evaluation outputs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp


def paired_t_test(left, right):
    """Pair by user ID, then test whether the mean of the differences is zero."""
    if left.keys() != right.keys() or len(left) < 2:
        raise ValueError("Need at least two identical user IDs on both sides")
    differences = np.array([left[user] - right[user] for user in sorted(left)], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Scores must be finite")
    result = {"users": len(differences), "mean_difference": float(differences.mean())}
    if np.all(differences == differences[0]):
        return {**result, "statistic": None, "p_value": 1.0 if differences[0] == 0 else None}
    statistic, p_value = ttest_1samp(differences, popmean=0, alternative="two-sided")
    return {**result, "statistic": float(statistic), "p_value": float(p_value)}


def load_scores(directories, reference):
    if len(set(path.resolve() for path in directories)) != len(directories):
        raise ValueError("A run must occur only once per model")
    all_scores = []
    for directory in directories:
        report = json.loads((directory / "metrics.json").read_bytes())
        signature = (report["split"], report["inputs"]["split"]["sha256"],
                     report["inputs"]["manifest"]["sha256"], report["protocol"]["exclude_seen"])
        if signature[-1] is not False:
            raise ValueError(f"{directory}: expected the shared protocol with seen items kept")
        if reference is not None and signature != reference:
            raise ValueError(f"{directory}: split, manifest or protocol differs")
        reference = signature
        scores = {}
        with (directory / "per_user.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                user, score = int(row["user_id"]), float(row["ndcg@10"])
                if user < 1 or user in scores or not 0 <= score <= 1:
                    raise ValueError(f"{directory}: invalid user ID or NDCG@10")
                scores[user] = score
        if len(scores) != report["counts"]["users"] or (all_scores and scores.keys() != all_scores[0].keys()):
            raise ValueError(f"{directory}: per-user data differs from the expected cohort")
        all_scores.append(scores)
    return {user: float(np.mean([run[user] for run in all_scores])) for user in all_scores[0]}, reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, nargs="+", required=True, help="Evaluation directories for the first model")
    parser.add_argument("--right", type=Path, nargs="+", required=True, help="Evaluation directories for the second model")
    args = parser.parse_args(argv)
    try:
        left, reference = load_scores(args.left, None)
        right, _ = load_scores(args.right, reference)
        result = paired_t_test(left, right)
        print(json.dumps({"split": reference[0], "metric": "ndcg@10", **result}, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

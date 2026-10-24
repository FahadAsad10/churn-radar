"""Generate data, train and compare models, and save the best one.

Usage: python -m churn_radar.train [--rows 7000] [--out models]
"""

import argparse
from pathlib import Path

from churn_radar.data import generate
from churn_radar.model import save, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/customers.csv", index=False)

    pipe, report = train(df, args.seed)
    save(pipe, report, args.out)

    print(f"Churn rate: {report.churn_rate:.1%} over {len(df)} customers")
    for name, auc in sorted(report.cv_auc.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} CV ROC-AUC {auc:.3f}")
    print(f"Best: {report.best_model}, threshold {report.threshold:.2f}")
    print("Test:", ", ".join(f"{k} {v:.3f}" for k, v in report.test.items()))
    print("Top drivers:", ", ".join(r["feature"] for r in report.importance[:4]))


if __name__ == "__main__":
    main()

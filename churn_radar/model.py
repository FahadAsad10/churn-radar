"""Training, evaluation and scoring for the churn model."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_radar.data import CATEGORICAL, NUMERIC, TARGET

FEATURES = NUMERIC + CATEGORICAL


def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def candidates(seed: int = 42) -> dict:
    # No class weighting: probabilities stay calibrated (they feed revenue-at-risk),
    # and class imbalance is handled by tuning the decision threshold instead.
    return {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, random_state=seed
        ),
    }


def best_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 on held-out data, instead of a fixed 0.5."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


@dataclass
class Report:
    best_model: str
    threshold: float
    cv_auc: dict
    test: dict
    confusion: list
    roc: dict
    importance: list
    n_train: int
    n_test: int
    churn_rate: float


def train(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, Report]:
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = {
        name: float(cross_val_score(build_pipeline(est), X_train, y_train, cv=cv, scoring="roc_auc").mean())
        for name, est in candidates(seed).items()
    }
    best_name = max(cv_auc, key=cv_auc.get)
    pipe = build_pipeline(candidates(seed)[best_name]).fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    threshold = best_threshold(y_test, proba)
    pred = (proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    step = max(1, len(fpr) // 100)

    imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1)
    importance = sorted(
        ({"feature": f, "importance": float(m)} for f, m in zip(FEATURES, imp.importances_mean, strict=True)),
        key=lambda r: r["importance"],
        reverse=True,
    )

    report = Report(
        best_model=best_name,
        threshold=round(threshold, 4),
        cv_auc={k: round(v, 4) for k, v in cv_auc.items()},
        test={
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
        },
        confusion=confusion_matrix(y_test, pred).tolist(),
        roc={"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()},
        importance=importance,
        n_train=len(X_train),
        n_test=len(X_test),
        churn_rate=round(float(y.mean()), 4),
    )
    return pipe, report


def save(pipe: Pipeline, report: Report, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(asdict(report), indent=2))


def load(out_dir: Path) -> tuple[Pipeline, dict]:
    return joblib.load(out_dir / "model.joblib"), json.loads((out_dir / "metrics.json").read_text())


def risk_band(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


def score(pipe: Pipeline, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["churn_probability"] = pipe.predict_proba(df[FEATURES])[:, 1].round(4)
    out["will_churn"] = (out["churn_probability"] >= threshold).astype(int)
    out["risk"] = out["churn_probability"].map(risk_band)
    return out.sort_values("churn_probability", ascending=False)

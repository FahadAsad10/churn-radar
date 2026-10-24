import pandas as pd
import pytest

from churn_radar.data import generate
from churn_radar.model import FEATURES, load, risk_band, save, score, train


@pytest.fixture(scope="module")
def trained():
    return train(generate(2500, seed=11), seed=11)


def test_model_beats_baseline(trained):
    _, report = trained
    assert report.test["roc_auc"] > 0.78
    assert report.best_model in report.cv_auc
    assert 0 < report.threshold < 1
    assert sum(map(sum, report.confusion)) == report.n_test


def test_contract_is_a_top_driver(trained):
    _, report = trained
    top = [r["feature"] for r in report.importance[:3]]
    assert "contract" in top


def test_score_ranks_and_bands(trained):
    pipe, report = trained
    df = generate(100, seed=99).drop(columns=["churned"])
    out = score(pipe, df, report.threshold)
    assert out["churn_probability"].is_monotonic_decreasing
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk"]) <= {"Low", "Medium", "High"}


def test_score_rejects_missing_columns(trained):
    pipe, report = trained
    with pytest.raises(ValueError, match="contract"):
        score(pipe, pd.DataFrame([{"tenure_months": 3}]), report.threshold)


def test_retention_lever_lowers_risk(trained):
    pipe, _ = trained
    risky = pd.DataFrame(
        [
            {
                "tenure_months": 3,
                "monthly_charges": 95.0,
                "total_charges": 285.0,
                "support_tickets": 4,
                "logins_last_30d": 3,
                "age": 25,
                "contract": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber",
                "tech_support": "No",
                "streaming": "Yes",
                "paperless_billing": "Yes",
            }
        ]
    )[FEATURES]
    before = pipe.predict_proba(risky)[0, 1]
    after = pipe.predict_proba(risky.assign(contract="Two year"))[0, 1]
    assert before > 0.6
    assert after < before


def test_save_and_load_roundtrip(trained, tmp_path):
    pipe, report = trained
    save(pipe, report, tmp_path)
    loaded, metrics = load(tmp_path)
    df = generate(20, seed=5)
    assert (loaded.predict_proba(df[FEATURES]) == pipe.predict_proba(df[FEATURES])).all()
    assert metrics["best_model"] == report.best_model


def test_risk_band():
    assert risk_band(0.1) == "Low"
    assert risk_band(0.45) == "Medium"
    assert risk_band(0.8) == "High"

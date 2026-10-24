from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_errors():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.startswith("📡 Churn Radar")
    labels = [m.label for m in at.metric]
    assert "Monthly revenue at risk" in labels
    assert "ROC-AUC" in labels

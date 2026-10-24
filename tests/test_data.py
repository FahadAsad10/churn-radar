from churn_radar.data import CATEGORICAL, NUMERIC, TARGET, generate


def test_shape_and_columns():
    df = generate(500, seed=1)
    assert len(df) == 500
    assert set(NUMERIC + CATEGORICAL + [TARGET, "customer_id"]) == set(df.columns)
    assert df.isna().sum().sum() == 0


def test_deterministic():
    assert generate(200, seed=7).equals(generate(200, seed=7))


def test_realistic_ranges_and_relationships():
    df = generate(5000, seed=3)
    assert df["tenure_months"].between(1, 72).all()
    assert df["monthly_charges"].between(18, 120).all()
    assert 0.15 < df[TARGET].mean() < 0.45
    by_contract = df.groupby("contract")[TARGET].mean()
    assert by_contract["Month-to-month"] > by_contract["One year"] > by_contract["Two year"]
    no_internet = df[df["internet_service"] == "None"]
    assert (no_internet["streaming"] == "No").all()

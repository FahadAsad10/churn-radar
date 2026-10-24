"""Synthetic but realistic subscription-customer dataset.

Churn depends on the features the way it does in real telecom and SaaS data:
month-to-month contracts, short tenure, high bills, support tickets and
electronic-check payments raise risk; long contracts, autopay and bundled
services lower it. A noise term keeps the problem from being trivially separable.
"""

import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "logins_last_30d", "age"]
CATEGORICAL = ["contract", "payment_method", "internet_service", "tech_support", "streaming", "paperless_billing"]
TARGET = "churned"


def generate(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    # Longer contracts go with longer-standing customers
    shape = np.select([contract == "Month-to-month", contract == "One year"], [1.2, 2.2], 3.5)
    scale = np.select([contract == "Month-to-month", contract == "One year"], [11.0, 12.0], 12.0)
    tenure = rng.gamma(shape, scale)
    tenure = np.where(tenure > 72, rng.uniform(48, 72, n), tenure)  # resample the long tail, no pile-up at 72
    tenure = np.clip(tenure, 1, 72).astype(int)

    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.35, 0.20])
    streaming = np.where(internet == "None", "No", rng.choice(["Yes", "No"], n, p=[0.45, 0.55]))
    tech_support = np.where(internet == "None", "No", rng.choice(["Yes", "No"], n, p=[0.35, 0.65]))
    payments = ["Electronic check", "Credit card (auto)", "Bank transfer (auto)", "Mailed check"]
    payment = rng.choice(payments, n, p=[0.35, 0.25, 0.22, 0.18])
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])

    base = np.select([internet == "Fiber", internet == "DSL"], [70.0, 50.0], 22.0)
    monthly = base + (streaming == "Yes") * 18 + (tech_support == "Yes") * 8 + rng.normal(0, 6, n)
    monthly = np.round(np.clip(monthly, 18, 120), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    tickets = rng.poisson(np.where(internet == "Fiber", 1.4, 0.8))
    logins = np.clip(rng.normal(18 - tickets * 1.5, 7, n), 0, 60).astype(int)
    age = np.clip(rng.normal(42, 14, n), 18, 85).astype(int)

    logit = (
        -1.1
        + 1.5 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.045 * tenure
        + 0.022 * (monthly - 60)
        + 0.38 * tickets
        - 0.05 * logins
        + 0.7 * (payment == "Electronic check")
        - 0.3 * np.isin(payment, ["Credit card (auto)", "Bank transfer (auto)"])
        + 0.45 * (internet == "Fiber")
        - 0.55 * (tech_support == "Yes")
        + 0.25 * (paperless == "Yes")
        + 0.012 * np.maximum(0, 30 - age)
        + rng.normal(0, 0.6, n)
    )
    churned = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "logins_last_30d": logins,
            "age": age,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "tech_support": tech_support,
            "streaming": streaming,
            "paperless_billing": paperless,
            TARGET: churned,
        }
    )


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customers.csv", index=False)
    print(f"Wrote {len(df)} rows, churn rate {df[TARGET].mean():.1%}")

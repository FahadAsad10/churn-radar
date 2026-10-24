"""Churn Radar dashboard: explore customers, inspect the model, and score new data."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churn_radar.data import CATEGORICAL, TARGET, generate
from churn_radar.model import FEATURES, load, risk_band, save, score, train

MODEL_DIR = Path("models")
ACCENT = "#8b7cff"
PALETTE = ["#8b7cff", "#4fd1c5", "#f6ad55", "#fc8181"]

st.set_page_config(page_title="Churn Radar", page_icon="📡", layout="wide")


@st.cache_resource
def get_model():
    if not (MODEL_DIR / "model.joblib").exists():
        with st.spinner("Training the model for the first time…"):
            save(*train(generate()), MODEL_DIR)
    return load(MODEL_DIR)


@st.cache_data
def get_data() -> pd.DataFrame:
    return generate()


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9c9d6"),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(gridcolor="#24242f")
    fig.update_yaxes(gridcolor="#24242f")
    return fig


pipe, metrics = get_model()
df = get_data()
threshold = metrics["threshold"]

st.title("📡 Churn Radar")
st.caption(
    f"Predicts which subscription customers are about to leave. Best model: **{metrics['best_model']}**, "
    f"test ROC-AUC **{metrics['test']['roc_auc']:.3f}**, decision threshold **{threshold:.2f}**."
)

overview, model_tab, predict_tab, batch_tab = st.tabs(["Overview", "Model", "Predict a customer", "Score a CSV"])

with overview:
    scored = score(pipe, df, threshold)
    at_risk = scored[scored["risk"] == "High"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Historical churn rate", f"{df[TARGET].mean():.1%}")
    c3.metric("High-risk customers", f"{len(at_risk):,}")
    c4.metric("Monthly revenue at risk", f"${at_risk['monthly_charges'].sum():,.0f}")

    left, right = st.columns(2)
    by_contract = df.groupby("contract", as_index=False)[TARGET].mean()
    fig = px.bar(by_contract, x="contract", y=TARGET, title="Churn rate by contract", color_discrete_sequence=[ACCENT])
    fig.update_yaxes(tickformat=".0%", title=None)
    left.plotly_chart(style(fig), use_container_width=True)

    fig = px.histogram(
        df.assign(status=df[TARGET].map({0: "Stayed", 1: "Churned"})),
        x="tenure_months",
        color="status",
        nbins=36,
        barmode="overlay",
        opacity=0.75,
        title="Tenure: churners leave early",
        color_discrete_map={"Stayed": PALETTE[1], "Churned": PALETTE[3]},
    )
    right.plotly_chart(style(fig), use_container_width=True)

    st.subheader("Highest-risk customers")
    st.dataframe(
        at_risk[["customer_id", "churn_probability", "contract", "tenure_months", "monthly_charges", "support_tickets"]]
        .head(10)
        .style.format({"churn_probability": "{:.0%}", "monthly_charges": "${:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

with model_tab:
    t = metrics["test"]
    cols = st.columns(5)
    for col, (label, key) in zip(
        cols,
        [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")],
        strict=True,
    ):
        col.metric(label, f"{t[key]:.3f}")

    left, right = st.columns(2)
    cv = pd.DataFrame(metrics["cv_auc"].items(), columns=["model", "cv_auc"]).sort_values("cv_auc")
    fig = px.bar(cv, x="cv_auc", y="model", orientation="h", title="5-fold CV ROC-AUC by model", text_auto=".3f")
    fig.update_traces(marker_color=[ACCENT if m == metrics["best_model"] else "#3a3a4a" for m in cv["model"]])
    fig.update_xaxes(range=[0.5, 1], title=None)
    fig.update_yaxes(title=None)
    left.plotly_chart(style(fig), use_container_width=True)

    roc = metrics["roc"]
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=ACCENT, width=3), name="Model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#555", dash="dash"), name="Random")
    fig.update_layout(title=f"ROC curve (AUC {t['roc_auc']:.3f})", xaxis_title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    right.plotly_chart(style(fig), use_container_width=True)

    left, right = st.columns(2)
    imp = pd.DataFrame(metrics["importance"]).sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h", title="What drives churn (permutation importance)")
    fig.update_traces(marker_color=ACCENT)
    fig.update_yaxes(title=None)
    left.plotly_chart(style(fig, 380), use_container_width=True)

    (tn, fp), (fn, tp) = metrics["confusion"]
    fig = px.imshow(
        [[tn, fp], [fn, tp]],
        x=["Predicted stay", "Predicted churn"],
        y=["Actually stayed", "Actually churned"],
        text_auto=True,
        color_continuous_scale=["#15151d", ACCENT],
        title="Confusion matrix on the test set",
    )
    fig.update_coloraxes(showscale=False)
    right.plotly_chart(style(fig, 380), use_container_width=True)

with predict_tab:
    st.write("Describe a customer to get their churn risk, and see which changes would lower it the most.")
    with st.form("customer"):
        a, b, c = st.columns(3)
        customer = {
            "tenure_months": a.slider("Tenure (months)", 1, 72, 8),
            "monthly_charges": a.slider("Monthly charges ($)", 18.0, 120.0, 79.0),
            "support_tickets": b.slider("Support tickets", 0, 8, 2),
            "logins_last_30d": b.slider("Logins in the last 30 days", 0, 60, 12),
            "age": c.slider("Age", 18, 85, 29),
        }
        customer["total_charges"] = customer["tenure_months"] * customer["monthly_charges"]
        options = {col: sorted(df[col].unique()) for col in CATEGORICAL}
        defaults = {"contract": "Month-to-month", "payment_method": "Electronic check", "internet_service": "Fiber"}
        for i, col in enumerate(CATEGORICAL):
            target = [a, b, c][i % 3]
            index = options[col].index(defaults[col]) if col in defaults else 0
            customer[col] = target.selectbox(col.replace("_", " ").capitalize(), options[col], index=index)
        st.form_submit_button("Predict", type="primary")

    row = pd.DataFrame([customer])[FEATURES]
    p = float(pipe.predict_proba(row)[0, 1])
    left, right = st.columns([1, 1.4])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 30], "color": "#15251f"},
                    {"range": [30, 60], "color": "#2a2416"},
                    {"range": [60, 100], "color": "#2a1618"},
                ],
            },
            title={"text": f"{risk_band(p)} risk"},
        )
    )
    left.plotly_chart(style(fig, 300), use_container_width=True)

    levers = {
        "Move to a two-year contract": {"contract": "Two year"},
        "Switch to autopay by card": {"payment_method": "Credit card (auto)"},
        "Add tech support": {"tech_support": "Yes"} if customer["internet_service"] != "None" else None,
        "Resolve open tickets (0 tickets)": {"support_tickets": 0},
        "Offer a $15 discount": {"monthly_charges": max(18.0, customer["monthly_charges"] - 15)},
    }
    rows = []
    for name, change in levers.items():
        if not change or all(customer[k] == v for k, v in change.items()):
            continue
        new_p = float(pipe.predict_proba(row.assign(**change))[0, 1])
        rows.append({"Retention lever": name, "New risk": new_p, "Change": new_p - p})
    if rows:
        right.markdown("**Retention levers**, ranked by how much each one lowers this customer's risk")
        right.dataframe(
            pd.DataFrame(rows).sort_values("Change").style.format({"New risk": "{:.0%}", "Change": "{:+.0%}"}),
            use_container_width=True,
            hide_index=True,
        )

with batch_tab:
    st.write(f"Upload a CSV with these columns: `{', '.join(FEATURES)}`. You get every customer ranked by risk.")
    sample = df.drop(columns=[TARGET]).sample(25, random_state=1)
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_customers.csv", "text/csv")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        try:
            result = score(pipe, pd.read_csv(upload), threshold)
        except ValueError as err:
            st.error(str(err))
        else:
            counts = result["risk"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("High risk", int(counts.get("High", 0)))
            c2.metric("Medium risk", int(counts.get("Medium", 0)))
            c3.metric("Low risk", int(counts.get("Low", 0)))
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", result.to_csv(index=False), "scored_customers.csv", "text/csv")

"""Résumé des runs de l'expérience cancer-sein."""

import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

r = mlflow.search_runs(
    experiment_names=["cancer-sein"],
    order_by=["attributes.start_time ASC"]
)

r = r.reindex(columns=[
    "tags.mlflow.runName",
    "tags.mlflow.parentRunId",
    "params.C",
    "params.logisticregression__C",
    "metrics.recall",
    "metrics.accuracy",
    "metrics.mean_test_score",
    "metrics.best_cv_score"
])

r.columns = [
    "run", "enfant", "C", "C_e",
    "recall", "accuracy", "recall_cv", "best_cv"
]

r["C"] = r["C"].fillna(r["C_e"]).astype(float)
r["recall_cv"] = r["recall_cv"].fillna(r["best_cv"]).round(4)

e = r["enfant"].notna()
r.loc[e, "run"] = "enfant"

r = pd.concat([
    r[~e],
    r[e].sort_values("C")
]).drop(columns=["C_e", "best_cv"])

print(
    r.drop(columns="enfant").to_string(
        index=False,
        na_rep=""
    )
)
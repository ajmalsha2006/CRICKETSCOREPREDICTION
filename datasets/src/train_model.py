from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DELIVERIES = ROOT / "deliveries.csv"
MATCHES = ROOT / "matches.csv"
OUT = ROOT / "outputs"
CATEGORICAL = ["batting_team", "bowling_team", "venue", "innings"]
NUMERIC = ["over_number", "score", "wickets_lost", "balls_faced", "run_rate", "season_start"]
FEATURES = CATEGORICAL + NUMERIC


def main() -> None:
    OUT.mkdir(exist_ok=True)
    deliveries = pd.read_csv(DELIVERIES)
    matches = pd.read_csv(MATCHES)

    required = {"match_id", "inning", "batting_team", "bowling_team", "over", "total_runs", "is_wicket"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries.csv is missing required columns: {sorted(missing)}")
    match_required = {"id", "season", "venue"}
    missing = match_required - set(matches.columns)
    if missing:
        raise ValueError(f"matches.csv is missing required columns: {sorted(missing)}")

    # Match metadata known before play only; result and target columns are intentionally excluded.
    metadata = matches[["id", "season", "venue"]].rename(columns={"id": "match_id"})
    data = deliveries.merge(metadata, on="match_id", how="inner", validate="many_to_one")
    data["total_runs"] = pd.to_numeric(data["total_runs"], errors="coerce").fillna(0)
    data["is_wicket"] = pd.to_numeric(data["is_wicket"], errors="coerce").fillna(0)
    data["over"] = pd.to_numeric(data["over"], errors="coerce")
    data = data.dropna(subset=["over", "batting_team", "bowling_team"])
    data["season_start"] = pd.to_numeric(data["season"].astype(str).str[:4], errors="coerce")
    data["season_start"] = data["season_start"].fillna(data["season_start"].median())
    data["innings"] = data["inning"].astype(str)
    data["venue"] = data["venue"].fillna("Unknown").astype(str)

    keys = ["match_id", "inning"]
    # One training example per observed over: state at the end of that over.
    by_over = (data.groupby(keys + ["over", "batting_team", "bowling_team", "venue", "season_start"],
                            as_index=False, dropna=False)
                  .agg(over_runs=("total_runs", "sum"), over_wickets=("is_wicket", "sum"),
                       balls_in_over=("total_runs", "size")))
    by_over["innings"] = by_over["inning"].astype(str)
    by_over = by_over.sort_values(keys + ["over"], kind="stable")
    by_over["score"] = by_over.groupby(keys)["over_runs"].cumsum()
    by_over["wickets_lost"] = by_over.groupby(keys)["over_wickets"].cumsum().clip(upper=10)
    by_over["balls_faced"] = by_over.groupby(keys)["balls_in_over"].cumsum()
    by_over["over_number"] = by_over["over"].astype(float) + 1
    by_over["run_rate"] = by_over["score"] / by_over["over_number"].clip(lower=1)
    totals = data.groupby(keys, as_index=False).agg(final_score=("total_runs", "sum"))
    samples = by_over.merge(totals, on=keys, how="inner", validate="many_to_one")
    samples = samples.dropna(subset=FEATURES + ["final_score"])
    samples["innings"] = samples["innings"].astype(str)

    match_ids = samples["match_id"].drop_duplicates().to_numpy()
    if len(match_ids) < 5:
        raise ValueError(f"Need at least 5 matches to create a holdout set; found {len(match_ids)}")
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
    train = samples[samples["match_id"].isin(train_ids)]
    test = samples[samples["match_id"].isin(test_ids)]

    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numbers", "passthrough", NUMERIC),
    ])
    model = Pipeline([
        ("features", prep),
        ("regressor", RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                                            max_features=0.9, n_jobs=-1, random_state=42)),
    ])
    model.fit(train[FEATURES], train["final_score"])
    predicted = model.predict(test[FEATURES])
    metrics = {
        "task": "predict final innings score from the state at an over break",
        "matches_total": int(len(match_ids)),
        "matches_train": int(len(train_ids)),
        "matches_test": int(len(test_ids)),
        "training_examples": int(len(train)),
        "test_examples": int(len(test)),
        "mae_runs": round(float(mean_absolute_error(test["final_score"], predicted)), 3),
        "rmse_runs": round(float(np.sqrt(mean_squared_error(test["final_score"], predicted))), 3),
        "r2": round(float(r2_score(test["final_score"], predicted)), 4),
        "split": "match-level holdout (20%, random_state=42)",
        "features": FEATURES,
        "note": "Inputs are over-break score, wickets, overs, teams, venue and season. Match outcome fields are not used.",
    }
    joblib.dump({"pipeline": model, "features": FEATURES}, OUT / "score_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_results = test[["match_id", "inning", "over_number", "score", "wickets_lost", "final_score"]].copy()
    test_results["predicted_final_score"] = np.round(predicted, 1)
    test_results.to_csv(OUT / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and evaluation files in: {OUT}")


if __name__ == "__main__":
    main()



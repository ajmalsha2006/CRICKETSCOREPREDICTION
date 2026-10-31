from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "first_innings_score_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
MATCHES_PARQUET = ROOT / "datasets" / "processed" / "matches.parquet"
MATCHES_CSV = ROOT / "datasets" / "matches.csv"

st.set_page_config(page_title="Cricket Score Predictor", page_icon="🏏", layout="wide")
st.title("🏏 Cricket Score Predictor")
st.caption("Make first-innings estimates and compare model training runs.")

if not MODEL_PATH.exists():
    st.error("The trained model was not found. Run the training notebook first.")
    st.stop()

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def tracking_uri():
    db_path = (ROOT / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{db_path}"


@st.cache_data(ttl=20)
def load_mlflow_runs(uri: str):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("Cricket first-innings score prediction")
    if experiment is None:
        return pd.DataFrame(), {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    records = []
    params_by_run = {}
    for run in runs:
        run_id = run.info.run_id
        params_by_run[run_id] = dict(run.data.params)
        records.append({
            "Run": run.data.tags.get("mlflow.runName", run_id[:8]),
            "Run ID": run_id,
            "Started": pd.to_datetime(run.info.start_time, unit="ms", errors="coerce"),
            "Status": run.info.status,
            **run.data.metrics,
        })
    return pd.DataFrame(records), params_by_run


@st.cache_data
def load_match_options():
    # DuckDB reads the compact Parquet match table; use the CSV if Parquet has not been created yet.
    try:
        import duckdb
        source = MATCHES_PARQUET if MATCHES_PARQUET.exists() else MATCHES_CSV
        con = duckdb.connect(database=":memory:")
        matches = con.execute("SELECT * FROM read_parquet(?)" if source.suffix == ".parquet" else "SELECT * FROM read_csv_auto(?)", [str(source)]).df()
        con.close()
    except Exception:
        matches = pd.read_csv(MATCHES_CSV)

    teams = sorted(set(matches.get("team1", pd.Series(dtype=str)).dropna().astype(str)) | set(matches.get("team2", pd.Series(dtype=str)).dropna().astype(str)))
    cities = sorted(value for value in matches.get("city", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if value not in {"NA", "Unknown"})
    venues = sorted(matches.get("venue", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    seasons = pd.to_numeric(matches.get("season", pd.Series(dtype=str)).astype(str).str.extract(r"(\d{4})")[0], errors="coerce").dropna().astype(int).unique().tolist()
    return teams, cities, venues, sorted(seasons)

try:
    artifact = load_model(str(MODEL_PATH))
    teams, cities, venues, seasons = load_match_options()
except Exception as exc:
    st.exception(exc)
    st.stop()

pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
late_classifier = artifact.get("late_innings_classifier")
late_min_overs = int(artifact.get("late_classifier_min_overs_elapsed", 16))
late_threshold = int(artifact.get("late_classifier_score_threshold", 180))

if not teams or not venues or not seasons:
    st.error("Team, venue, or season options could not be loaded from the project datasets.")
    st.stop()

predict_tab, compare_tab = st.tabs(["Score Predictor", "MLflow Run Comparison"])

with predict_tab:
    st.subheader("Match situation")
    with st.form("prediction_form"):
        first, second = st.columns(2)
        with first:
            batting_team = st.selectbox("Batting team", teams)
            current_score = st.number_input("Current score", min_value=0, max_value=400, value=80, step=1)
            wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=2, step=1)
            overs_elapsed = st.number_input("Completed overs", min_value=1, max_value=19, value=10, step=1)
            balls_faced = st.number_input("Deliveries faced", min_value=1, max_value=120, value=60, step=1, help="Usually six per completed over. Adjust this if wides or no-balls changed the delivery count.")
        with second:
            bowling_team = st.selectbox("Bowling team", teams, index=min(1, len(teams) - 1))
            season = st.selectbox("Season", seasons, index=len(seasons) - 1)
            city = st.selectbox("City", ["Unknown"] + cities)
            venue = st.selectbox("Venue", venues)
        predict = st.form_submit_button("Predict first-innings total", type="primary", use_container_width=True)

    if predict:
        if batting_team == bowling_team:
            st.error("Choose different batting and bowling teams.")
        elif balls_faced < overs_elapsed * 6:
            st.warning("Deliveries faced is below six per completed over. Check the match state before using this estimate.")
        else:
            row = pd.DataFrame([{
                "season": season,
                "overs_elapsed": overs_elapsed,
                "current_score": current_score,
                "wickets_lost": wickets_lost,
                "balls_faced": balls_faced,
                "run_rate": current_score / overs_elapsed,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "venue": venue,
            }], columns=feature_columns)
            estimate = max(int(current_score), int(round(float(pipeline.predict(row)[0]))))
            score_col, remaining_col, rate_col = st.columns(3)
            score_col.metric("Estimated first-innings total", f"{estimate} runs")
            remaining_col.metric("Estimated runs remaining", f"{max(0, estimate - int(current_score))} runs")
            rate_col.metric("Current run rate", f"{current_score / overs_elapsed:.2f}")

            if late_classifier is not None and overs_elapsed >= late_min_overs:
                probability = float(late_classifier.predict_proba(row)[0][1])
                st.metric(f"Chance of reaching {late_threshold}+ runs", f"{probability:.1%}")
            elif late_classifier is not None:
                st.info(f"The {late_threshold}+ score classifier is available after {late_min_overs} completed overs.")

            st.caption("This is a historical-data estimate. It is most useful as an approximate forecast, not a guaranteed final score.")

    with st.expander("Model evaluation"):
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            col1, col2, col3 = st.columns(3)
            col1.metric("Score error (MAE)", f"{metrics.get('MAE', float('nan')):.1f} runs")
            col2.metric("Late classifier accuracy", f"{metrics.get('late_classifier_accuracy', float('nan')):.1%}")
            col3.metric("Late classifier precision", f"{metrics.get('late_classifier_precision', float('nan')):.1%}")
            st.caption("Late classifier metrics apply to predictions made after at least 16 overs for whether the innings reaches 180 runs.")

with compare_tab:
    st.subheader("MLflow run comparison")
    st.caption("Compare score errors, 180+ classification metrics, and training settings from the local MLflow history.")
    mlflow_db = ROOT / "mlflow.db"
    if not mlflow_db.exists():
        st.info("No MLflow run history yet. Run the training notebook once to create it.")
    else:
        try:
            runs_df, params_by_run = load_mlflow_runs(tracking_uri())
        except Exception as exc:
            st.error("Could not read the local MLflow history.")
            st.exception(exc)
            runs_df, params_by_run = pd.DataFrame(), {}

        if runs_df.empty:
            st.info("No training runs found. Run the training notebook to record the first experiment.")
        else:
            runs_df = runs_df.sort_values("Started", ascending=False, na_position="last").reset_index(drop=True)
            summary_columns = [
                "Run", "Run ID", "Started", "Status", "MAE", "RMSE", "R2",
                "high_score_accuracy", "late_classifier_accuracy",
                "late_classifier_precision", "late_classifier_recall",
            ]
            available_columns = [column for column in summary_columns if column in runs_df.columns]
            st.dataframe(runs_df[available_columns], hide_index=True, use_container_width=True)

            labels = {
                f"{row['Run']} · {row['Run ID'][:8]}": row['Run ID']
                for _, row in runs_df.iterrows()
            }
            selected_labels = st.multiselect(
                "Choose runs to compare",
                options=list(labels),
                default=list(labels)[:min(3, len(labels))],
            )
            selected_ids = [labels[label] for label in selected_labels]
            selected = runs_df[runs_df["Run ID"].isin(selected_ids)].copy()
            if not selected.empty:
                selected["Run"] = selected.apply(lambda row: f"{row['Run']} · {row['Run ID'][:8]}", axis=1)
                left, right = st.columns(2)
                error_metrics = [name for name in ["MAE", "RMSE"] if name in selected.columns]
                classification_metrics = [name for name in ["late_classifier_accuracy", "late_classifier_precision", "late_classifier_recall"] if name in selected.columns]
                with left:
                    st.markdown("**Score prediction error**")
                    if error_metrics:
                        st.bar_chart(selected.set_index("Run")[error_metrics])
                with right:
                    st.markdown("**Late 180+ classifier**")
                    if classification_metrics:
                        st.bar_chart(selected.set_index("Run")[classification_metrics])

                chosen_label = st.selectbox("View run settings", selected_labels)
                chosen_run_id = labels[chosen_label]
                st.json(params_by_run.get(chosen_run_id, {}))


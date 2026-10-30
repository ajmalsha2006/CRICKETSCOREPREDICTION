# Cricket innings score prediction

This project trains a regression model to estimate an innings' final total from the match state at the end of each observed over. It uses the supplied `deliveries.csv` and `matches.csv` files.

## Setup

From this folder, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
python .\src\train_model.py
```

The script holds out 20% of matches (rather than deliveries) for evaluation and reports mean absolute error, root mean squared error, and R². It saves the fitted pipeline to `outputs/score_model.joblib`, metrics to `outputs/metrics.json`, and per-sample holdout predictions to `outputs/test_predictions.csv`.

## Model inputs

At each over break, inputs are the batting and bowling teams, venue, innings number, season, current score, wickets lost, deliveries faced, overs elapsed, and current run rate. Match results, winners, player of the match, and chase targets are excluded to avoid leaking information from after the prediction point.

This is a first baseline for score estimation. It predicts the final innings total from intermediate states; it does not predict a winner or a second-innings chase target.

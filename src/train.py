import numpy as np, joblib, yaml, mlflow, mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression

params = yaml.safe_load(open("params.yaml"))
cfg = params["model"]
exp_name = params["tracking"]["experiment"]
tracking_uri = params["tracking"]["tracking_uri"]

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(exp_name)

Xtr = np.load("data/processed/X_train.npy")
ytr = np.load("data/processed/y_train.npy")

with mlflow.start_run(run_name="train"):
    mlflow.log_params(cfg)

    clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"])
    clf.fit(Xtr, ytr)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/model.pkl")
    mlflow.sklearn.log_model(clf, artifact_path="model")  # versi MLflow
    mlflow.log_artifact("models/model.pkl")

    mlflow.set_tag("model_type", params["model"]["type"])
    print("Model trained & logged to MLflow")
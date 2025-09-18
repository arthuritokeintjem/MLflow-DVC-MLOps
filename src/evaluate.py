import numpy as np, json, mlflow, yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

params = yaml.safe_load(open("params.yaml"))
tracking_uri = params["tracking"]["tracking_uri"]
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(params["tracking"]["experiment"])

Xte = np.load("data/processed/X_test.npy")
yte = np.load("data/processed/y_test.npy")

# ambil model dari file lokal models/
import joblib
clf = joblib.load("models/model.pkl")
yp = clf.predict(Xte)
acc = float(accuracy_score(yte, yp))
cm  = confusion_matrix(yte, yp)

Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f, indent=2)

# plot confusion matrix
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/confusion_matrix.png", bbox_inches="tight")
plt.close()

with mlflow.start_run(run_name="evaluate"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifact("reports/metrics.json")
    mlflow.log_artifact("reports/confusion_matrix.png")
print("Evaluation done. Acc=", acc)
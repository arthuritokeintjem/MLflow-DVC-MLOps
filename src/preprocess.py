import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))
test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]

df = pd.read_csv("data/raw/iris.csv")
X = df.drop(columns=["species"]).values.astype(float)
y = df["species"].astype("category").cat.codes.values  # 0/1/2

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

Path("data/processed").mkdir(parents=True, exist_ok=True)
np.save("data/processed/X_train.npy", Xtr)
np.save("data/processed/X_test.npy",  Xte)
np.save("data/processed/y_train.npy", ytr)
np.save("data/processed/y_test.npy",  yte)
print("Preprocess done:", Xtr.shape, Xte.shape)
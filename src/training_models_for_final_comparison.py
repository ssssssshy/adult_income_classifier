import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb


data_path = "../data/processed/adult_balanced.csv"
df = pd.read_csv(data_path)

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


def evaluate_model(y_true, y_pred, name=""):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators": [100],
        "max_depth": [5],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    model = xgb.XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def train_lightgbm(X_train, y_train):
    param_grid = {
        "n_estimators": [100],
        "max_depth": [5],
        "learning_rate": [0.1],
        "num_leaves": [31],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    model = lgb.LGBMClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_eval_nn(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dim=64,
    lr=0.001,
    batch_size=64,
    epochs=10,
    dropout=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    model = SimpleNN(X_train.shape[1], hidden_dim, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        preds = model(X_val_t).cpu().numpy()
        preds_labels = (preds > 0.5).astype(int)

    f1 = f1_score(y_val.values, preds_labels)
    return model, preds_labels, f1


def search_best_nn(X_train, y_train, X_val, y_val):
    param_grid = {
        "hidden_dim": [64],
        "lr": [0.001],
        "batch_size": [64],
        "dropout": [0.5],
    }

    best_model, best_f1, best_params = None, 0, None
    for params in ParameterGrid(param_grid):
        model, preds, f1 = train_eval_nn(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_dim=params["hidden_dim"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            dropout=params["dropout"],
        )
        if f1 > best_f1:
            best_model = model
            best_f1 = f1
            best_params = params

    return best_model, best_params


def save_model(model, name, model_type="sklearn"):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pkl" if model_type == "sklearn" else f"models/{name}.pt"

    if model_type == "sklearn":
        joblib.dump(model, path)
    elif model_type == "torch":
        torch.save(model.state_dict(), path)
    print(f"✅ Saved: {path}")


print("\n🔧 Тренировка моделей...")

xgb_model, xgb_params = train_xgboost(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_metrics = evaluate_model(y_test, xgb_preds)

lgb_model, lgb_params = train_lightgbm(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_metrics = evaluate_model(y_test, lgb_preds)

nn_model, nn_params = search_best_nn(X_train, y_train, X_test, y_test)
nn_model.eval()
with torch.no_grad():
    X_test = X_test.astype(np.float32)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(
        next(nn_model.parameters()).device
    )
    nn_preds = nn_model(X_test_tensor).cpu().numpy()
    nn_preds_labels = (nn_preds > 0.5).astype(int).flatten()

nn_metrics = evaluate_model(y_test, nn_preds_labels)

results = pd.DataFrame(
    {"XGBoost": xgb_metrics, "LightGBM": lgb_metrics, "NeuralNet": nn_metrics}
).T

print("\n📊 Сравнение моделей по метрикам:")
print(results)

save_model(xgb_model, "xgboost_model")
save_model(lgb_model, "lightgbm_model")
save_model(nn_model, "neural_net_model", model_type="torch")


print(X_train.dtypes)
print(X_train.head())

joblib.dump(xgb_model, "../models/xgboost_model.pkl")

joblib.dump(lgb_model, "../models/lightgbm_model.pkl")

torch.save(nn_model.state_dict(), "../models/neural_net_model.pt")

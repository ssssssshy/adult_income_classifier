import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


data_path = "../data/processed/adult_balanced.csv"
df = pd.read_csv(data_path)

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True) * 100}")
print(f"Test target distribution:\n{y_test.value_counts(normalize=True) * 100}")


def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return model, metrics


model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("Метрики LightGBM:", metrics)
    return model, metrics


model, metrics = train_lightgbm(X_train, y_train, X_test, y_test)
metrics


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch(X_train, y_train, X_test, y_test, epochs=20, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(
        y_train.values.reshape(-1, 1), dtype=torch.float32
    ).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(
        device
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor).cpu().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("Метрики PyTorch NN:", metrics)
    return model, metrics


model, metrics = train_pytorch(X_train, y_train, X_test, y_test)
metrics

# XGBoost — GridSearchCV
model = xgb.XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1],
    "colsample_bytree": [0.6, 0.8, 1],
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
print("Лучшие параметры XGBoost:", grid_search.best_params_)

# LightGBM— GridSearchCV

model = lgb.LGBMClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "num_leaves": [31, 50, 100],
    "subsample": [0.6, 0.8, 1],
    "colsample_bytree": [0.6, 0.8, 1],
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_lgbm = grid_search.best_estimator_
print("Лучшие параметры LightGBM:", grid_search.best_params_)


# neural network— GridSearchCV
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

    model = SimpleNN(X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(
        device
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        y_val_np = y_val.values
        preds = model(X_val_t).cpu().numpy()
        preds_labels = (preds > 0.5).astype(int)

    f1 = f1_score(y_val_np, preds_labels)
    return model, f1


X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


def hyperparam_search_nn(X_train, y_train, X_val, y_val):
    param_grid = {
        "hidden_dim": [32, 64, 128],
        "lr": [0.001, 0.0005],
        "batch_size": [32, 64],
        "dropout": [0.3, 0.5],
    }

    best_f1 = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        print(f"Тестируем параметры: {params}")
        model, f1 = train_eval_nn(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_dim=params["hidden_dim"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            dropout=params["dropout"],
            epochs=10,
        )
        print(f"F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

    print("\nЛучшие параметры:", best_params)
    print("Лучшее F1:", best_f1)
    return best_model, best_params, best_f1


model, best_params, best_f1 = hyperparam_search_nn(X_train, y_train, X_test, y_test)

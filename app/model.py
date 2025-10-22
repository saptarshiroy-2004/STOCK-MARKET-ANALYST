from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class TrainResult:
    model: RandomForestRegressor
    mae: float
    rmse: float


def train_return_model(X: pd.DataFrame, y: pd.Series) -> TrainResult:
    """
    Simple RandomForest regressor on return features.
    Uses last 20% as validation to report MAE/RMSE.
    """
    if len(X) < 200:
        # fall back to train on all with no validation if too small
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        return TrainResult(model=rf, mae=float("nan"), rmse=float("nan"))

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    preds = rf.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    return TrainResult(model=rf, mae=float(mae), rmse=float(rmse))

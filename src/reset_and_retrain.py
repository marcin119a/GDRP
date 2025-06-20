import torch
import torch.nn as nn
import torch.nn.init as init
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from src.models import MLPRegressor
from src.metrics import pearson_corr
from src.train_regression import train
from src.data_utils import load_data


def reset_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


def retrain_with_reset(clf, data_path="data/filtered_merged_df.parquet"):
    # Load data
    X_train, X_test, y_train, y_test, train_loader = load_data(data_path)

    # Preapre a model
    model_regression = MLPRegressor(clf)
    model_regression.apply(reset_weights)
    optimizer = torch.optim.Adam(model_regression.parameters(), lr=0.001)

    # Trening
    train(model_regression, train_loader, optimizer, epochs=50)

    # Evaluation
    model_regression.eval()
    with torch.no_grad():
        y_pred = model_regression(X_test).squeeze()
        mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
        rmse = mse**0.5
        pearson = pearson_corr(y_pred, y_test.squeeze()).item()
        spearman, _ = spearmanr(y_test.numpy().squeeze(), y_pred.numpy())

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Pearson Correlation: {pearson:.4f}")
    print(f"Spearman Correlation: {spearman:.4f}")
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
        print(f"RMSE: {np.sqrt(mse):.4f}")
        return y_pred

def plot_residuals(y_pred, y_test):
    residuals = y_test.numpy().squeeze() - y_pred.numpy()
    plt.scatter(y_pred.numpy(), residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted AUC")
    plt.ylabel("Residual")
    plt.title("Residuals vs Prediction")
    plt.show()
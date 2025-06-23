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


def plot_residuals(y_pred, y_test, text="Residuals vs Prediction", y_lim=None, x_lim=None):
    residuals = y_test.numpy().squeeze() - y_pred.numpy()
    predicted = y_pred.numpy()

    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted AUC")
    plt.ylabel("Residual")
    plt.title(text)

    if y_lim is not None:
        plt.ylim(y_lim)
    if x_lim is not None:
        plt.xlim(x_lim)

    plt.show()
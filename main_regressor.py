import torch
from torch import optim
from src.data_utils import load_data
from src.models import MLPClassifier, MLPRegressor
from src.train_regression import train
from src.evaluate import evaluate, plot_residuals
from src.metrics import pearson_corr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from src.utils import set_seed

set_seed(42)

# === Load data ===
X_train, X_test, y_train, y_test, train_loader = load_data("data/filtered_merged_df.parquet")

# === Load pretrained classifier ===
input_dim = X_train.shape[1]
num_classes = 24
clf = MLPClassifier(input_dim, num_classes)
clf.load_state_dict(torch.load("models/mlp_classifier_v_01.0.pt", map_location=torch.device('cpu')))

epochs = 50
learning_rate = 0.001

# === === I. Trening without reset weights === ===
print("=== Trening without reset weights ===")
reg = MLPRegressor(clf)
optimizer = optim.Adam(reg.parameters(), lr=learning_rate)
train(reg, train_loader, optimizer, epochs=epochs)

y_pred = evaluate(reg, X_test, y_test)
reg.eval()
with torch.no_grad():
    y_pred_reset = reg(X_test).squeeze()
    y_true = y_test.squeeze()
    mse = mean_squared_error(y_true.numpy(), y_pred_reset.numpy())
    rmse = mse**0.5
    pearson = pearson_corr(y_pred_reset, y_true).item()
    spearman_corr, _ = spearmanr(y_true.numpy(), y_pred_reset.numpy())

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Pearson Correlation: {pearson:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

from src.reset_and_retrain import reset_weights

print("\n=== Reset wag i ponowny trening ===")
reg_reset = MLPRegressor(clf)
reg_reset.apply(reset_weights)
optimizer_reset = optim.Adam(reg_reset.parameters(), lr=learning_rate)
train(reg_reset, train_loader, optimizer_reset, epochs=epochs)


# Evaluation
reg_reset.eval()
with torch.no_grad():
    y_pred_reset = reg_reset(X_test).squeeze()
    y_true = y_test.squeeze()
    mse = mean_squared_error(y_true.numpy(), y_pred_reset.numpy())
    rmse = mse**0.5
    pearson = pearson_corr(y_pred_reset, y_true).item()
    spearman_corr, _ = spearmanr(y_true.numpy(), y_pred_reset.numpy())

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Pearson Correlation: {pearson:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

residuals_1 = y_test.numpy().squeeze() - y_pred.numpy()
residuals_2 = y_test.numpy().squeeze() - y_pred_reset.numpy()

preds_1 = y_pred.numpy()
preds_2 = y_pred_reset.numpy()

min_resid = min(residuals_1.min(), residuals_2.min())
max_resid = max(residuals_1.max(), residuals_2.max())
y_lim = (min_resid, max_resid)

min_pred = min(preds_1.min(), preds_2.min())
max_pred = max(preds_1.max(), preds_2.max())
x_lim = (min_pred, max_pred)

plot_residuals(y_pred, y_test, text="GDRP Residuals – Sorafenib – Full Fine-Tuning", y_lim=y_lim, x_lim=x_lim)
plot_residuals(y_pred_reset, y_test, text="GDRP Residuals – Sorafenib – Reset Weights", y_lim=y_lim, x_lim=x_lim)
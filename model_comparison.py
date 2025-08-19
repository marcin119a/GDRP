import torch
from torch import optim
from src.data_utils import load_data
from src.models import MLPClassifier, MLPRegressor, LinearRegression, SimpleNN
from src.train_regression import train
from src.evaluate import evaluate, plot_residuals
from src.metrics import pearson_corr, spearman_corr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.utils import set_seed
from src.reset_and_retrain import reset_weights
import matplotlib.pyplot as plt
import numpy as np

set_seed(42)

# === Load data ===
print("=== Loading data ===")
X_train, X_test, y_train, y_test, train_loader = load_data("data/filtered_merged_df.parquet")

print(f"Data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}")

# === Load pretrained classifier ===
input_dim = X_train.shape[1]
num_classes = 24
clf = MLPClassifier(input_dim, num_classes)
clf.load_state_dict(torch.load("models/mlp_classifier_v_01.0.pt", map_location=torch.device('cpu')))

epochs = 50
learning_rate = 0.001

# === Model comparison results storage ===
results = {}

# === 1. Linear Regression (Sklearn) ===
print("\n=== 1. Linear Regression (Sklearn) ===")
lr_sklearn = SklearnLinearRegression()
lr_sklearn.fit(X_train.numpy(), y_train.numpy().flatten())
y_pred_lr = lr_sklearn.predict(X_test.numpy())

mse_lr = mean_squared_error(y_test.numpy().flatten(), y_pred_lr)
rmse_lr = mse_lr**0.5
r2_lr = r2_score(y_test.numpy().flatten(), y_pred_lr)
pearson_lr = pearson_corr(torch.tensor(y_pred_lr), y_test)
spearman_lr = spearman_corr(torch.tensor(y_pred_lr), y_test)

print(f"RMSE: {rmse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print(f"Pearson Correlation: {pearson_lr:.4f}")
print(f"Spearman Correlation: {spearman_lr:.4f}")

results['Linear Regression'] = {
    'y_pred': y_pred_lr,
    'rmse': rmse_lr,
    'r2': r2_lr,
    'pearson': pearson_lr,
    'spearman': spearman_lr
}

# === 2. Random Forest ===
print("\n=== 2. Random Forest ===")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train.numpy(), y_train.numpy().flatten())
y_pred_rf = rf.predict(X_test.numpy())

mse_rf = mean_squared_error(y_test.numpy().flatten(), y_pred_rf)
rmse_rf = mse_rf**0.5
r2_rf = r2_score(y_test.numpy().flatten(), y_pred_rf)
pearson_rf = pearson_corr(torch.tensor(y_pred_rf), y_test)
spearman_rf = spearman_corr(torch.tensor(y_pred_rf), y_test)

print(f"RMSE: {rmse_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")
print(f"Pearson Correlation: {pearson_rf:.4f}")
print(f"Spearman Correlation: {spearman_rf:.4f}")

results['Random Forest'] = {
    'y_pred': y_pred_rf,
    'rmse': rmse_rf,
    'r2': r2_rf,
    'pearson': pearson_rf,
    'spearman': spearman_rf
}

# === 3. Simple Neural Network ===
print("\n=== 3. Simple Neural Network ===")
simple_nn = SimpleNN(input_dim)
optimizer_nn = optim.Adam(simple_nn.parameters(), lr=learning_rate)
train(simple_nn, train_loader, optimizer_nn, epochs=epochs)

simple_nn.eval()
with torch.no_grad():
    y_pred_nn = simple_nn(X_test).squeeze()
    y_true = y_test.squeeze()
    mse_nn = mean_squared_error(y_true.numpy(), y_pred_nn.numpy())
    rmse_nn = mse_nn**0.5
    r2_nn = r2_score(y_true.numpy(), y_pred_nn.numpy())
    pearson_nn = pearson_corr(y_pred_nn, y_true)
    spearman_nn = spearman_corr(y_pred_nn, y_true)

print(f"RMSE: {rmse_nn:.4f}")
print(f"R² Score: {r2_nn:.4f}")
print(f"Pearson Correlation: {pearson_nn:.4f}")
print(f"Spearman Correlation: {spearman_nn:.4f}")

results['Simple Neural Network'] = {
    'y_pred': y_pred_nn.numpy(),
    'rmse': rmse_nn,
    'r2': r2_nn,
    'pearson': pearson_nn,
    'spearman': spearman_nn
}

# === 4. MLP Regressor (Reset Weights) ===
print("\n=== 4. MLP Regressor (Reset Weights) ===")
reg_reset = MLPRegressor(clf)
#reg_reset.apply(reset_weights)
optimizer_reset = optim.Adam(reg_reset.parameters(), lr=learning_rate)
train(reg_reset, train_loader, optimizer_reset, epochs=epochs)

reg_reset.eval()
with torch.no_grad():
    y_pred_mlp = reg_reset(X_test).squeeze()
    y_true = y_test.squeeze()
    mse_mlp = mean_squared_error(y_true.numpy(), y_pred_mlp.numpy())
    rmse_mlp = mse_mlp**0.5
    r2_mlp = r2_score(y_true.numpy(), y_pred_mlp.numpy())
    pearson_mlp = pearson_corr(y_pred_mlp, y_true)
    spearman_mlp = spearman_corr(y_pred_mlp, y_true)

print(f"RMSE: {rmse_mlp:.4f}")
print(f"R² Score: {r2_mlp:.4f}")
print(f"Pearson Correlation: {pearson_mlp:.4f}")
print(f"Spearman Correlation: {spearman_mlp:.4f}")

results['MLP Regressor (Reset)'] = {
    'y_pred': y_pred_mlp.numpy(),
    'rmse': rmse_mlp,
    'r2': r2_mlp,
    'pearson': pearson_mlp,
    'spearman': spearman_mlp
}

# === Summary Table ===
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"{'Model':<25} {'RMSE':<10} {'R²':<10} {'Pearson':<10} {'Spearman':<10}")
print("-"*80)
for model_name, metrics in results.items():
    print(f"{model_name:<25} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f} "
          f"{metrics['pearson']:<10.4f} {metrics['spearman']:<10.4f}")

# === Find best model for each metric ===
print("\n" + "="*60)
print("BEST MODELS BY METRIC")
print("="*60)
best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])
best_r2 = max(results.items(), key=lambda x: x[1]['r2'])
best_pearson = max(results.items(), key=lambda x: x[1]['pearson'])
best_spearman = max(results.items(), key=lambda x: x[1]['spearman'])

print(f"Best RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})")
print(f"Best R²: {best_r2[0]} ({best_r2[1]['r2']:.4f})")
print(f"Best Pearson: {best_pearson[0]} ({best_pearson[1]['pearson']:.4f})")
print(f"Best Spearman: {best_spearman[0]} ({best_spearman[1]['spearman']:.4f})")

# === Plotting comparison ===
y_true_np = y_test.numpy().flatten()

# Calculate global limits for consistent plotting
all_residuals = []
all_predictions = []
for model_name, metrics in results.items():
    residuals = y_true_np - metrics['y_pred']
    all_residuals.extend(residuals)
    all_predictions.extend(metrics['y_pred'])

min_resid = min(all_residuals)
max_resid = max(all_residuals)
y_lim = (min_resid, max_resid)

min_pred = min(all_predictions)
max_pred = max(all_predictions)
x_lim = (min_pred, max_pred)

# Plot residuals for each model
print("\n=== Generating residual plots ===")
for model_name, metrics in results.items():
    plot_residuals(torch.tensor(metrics['y_pred']), y_test, 
                   text=f"GDRP Residuals – Sorafenib – {model_name}", 
                   y_lim=y_lim, x_lim=x_lim)

# === Comparison plot of predictions vs actual ===
plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'red', 'green']
markers = ['o', 's', '^', 'D']

for i, (model_name, metrics) in enumerate(results.items()):
    plt.scatter(y_true_np, metrics['y_pred'], 
               alpha=0.6, label=f"{model_name} (R²={metrics['r2']:.3f})", 
               color=colors[i], marker=markers[i])

# Perfect prediction line
min_val = min(y_true_np.min(), min(all_predictions))
max_val = max(y_true_np.max(), max(all_predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

plt.xlabel('Actual AUC')
plt.ylabel('Predicted AUC')
plt.title('Model Comparison: Predicted vs Actual AUC Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === Bar chart comparison of metrics ===
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

models = list(results.keys())
rmse_values = [results[model]['rmse'] for model in models]
r2_values = [results[model]['r2'] for model in models]
pearson_values = [results[model]['pearson'] for model in models]
spearman_values = [results[model]['spearman'] for model in models]

# RMSE (lower is better)
bars1 = ax1.bar(models, rmse_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
ax1.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE')
ax1.tick_params(axis='x', rotation=45)
for bar, value in zip(bars1, rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# R² (higher is better)
bars2 = ax2.bar(models, r2_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
ax2.set_title('R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score')
ax2.tick_params(axis='x', rotation=45)
for bar, value in zip(bars2, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Pearson Correlation (higher is better)
bars3 = ax3.bar(models, pearson_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
ax3.set_title('Pearson Correlation Comparison (Higher is Better)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Pearson Correlation')
ax3.tick_params(axis='x', rotation=45)
for bar, value in zip(bars3, pearson_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Spearman Correlation (higher is better)
bars4 = ax4.bar(models, spearman_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
ax4.set_title('Spearman Correlation Comparison (Higher is Better)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Spearman Correlation')
ax4.tick_params(axis='x', rotation=45)
for bar, value in zip(bars4, spearman_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# === Feature importance for Random Forest ===
print("\n=== Random Forest Feature Importance ===")
feature_importance = rf.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
plt.yticks(range(len(top_features_idx)), [f'Feature {i}' for i in top_features_idx])
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features (Random Forest)', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# === Final summary ===
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Overall best performing model: {best_r2[0]}")
print(f"Best R² Score: {best_r2[1]['r2']:.4f}")
print(f"Best RMSE: {best_rmse[1]['rmse']:.4f}")
print(f"Best Pearson Correlation: {best_pearson[1]['pearson']:.4f}")
print(f"Best Spearman Correlation: {best_spearman[1]['spearman']:.4f}")

print("\nModel comparison completed! Check the plots above for visual analysis.")

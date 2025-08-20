import torch
from torch import optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
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
import os

set_seed(42)

def load_drug_data(drug_name, data_path="data/merged_df_all_drugs.parquet"):
    """
    Load data for a specific drug from the merged dataframe.
    """
    print(f"Loading data for drug: {drug_name}")
    
    # Load the full dataset
    df = pd.read_parquet(data_path)
    
    disease_list = [
        "Liver",
        "Soft tissue",
        "Thyroid",
        "Urinary tract",
        "Bone",
        "Stomach",
        "Pleura",
        "Ovary",
        "Kidney",
        "Large intestine",
        "Autonomic ganglia",
        "Breast",
        "Pancreas",
        "Upper aerodigestive tract",
        "Esophagus",
        "Central nervous system",
        "Skin",
        "Lung",
        "Hematopoietic and lymphoid tissue"
    ]
    mapping = {
        "Kidney Cancer": "Kidney",
        "Bladder Cancer": "Urinary tract",
        "Pancreatic Cancer": "Pancreas",
        "Colon/Colorectal Cancer": "Large intestine",
        "Breast Cancer": "Breast",
        "Ovarian Cancer": "Ovary",
        "Skin Cancer": "Skin",
        "Brain Cancer": "Central nervous system",
        "Lung Cancer": "Lung",
    }

    # Filter data for the specific drug
    drug_df = df[df['DRUG_NAME'] == drug_name].copy()
    drug_df['mapped_cancer'] = drug_df['primary_disease'].map(mapping)
    drug_df = drug_df[drug_df['mapped_cancer'].isin(disease_list)].dropna()
    
    if len(drug_df) == 0:
        raise ValueError(f"No data found for drug: {drug_name}")
    
    print(f"Found {len(drug_df)} samples for {drug_name}")
    
    # Create feature vector by concatenating TPM and symbol_counts
    drug_df['feature_vector'] = drug_df.apply(
        lambda row: np.concatenate([row['TPM'], row['symbol_counts']]),
        axis=1
    )
    
    # Extract features and target
    X = np.array(drug_df['feature_vector'].tolist())
    y = np.array(drug_df['AUC'])
    
    print(f"Feature vector dimension: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create data loader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    
    return X_train, X_test, y_train, y_test, train_loader

def get_available_drugs(data_path="data/merge_df_all_drugs.parquet"):
    """
    Get list of available drugs in the dataset.
    """
    df = pd.read_parquet(data_path)
    drugs = df['DRUG_NAME'].unique()
    print(f"Available drugs: {len(drugs)}")
    for i, drug in enumerate(sorted(drugs)):
        count = len(df[df['DRUG_NAME'] == drug])
        print(f"  {i+1:2d}. {drug:<20} ({count:4d} samples)")
    return sorted(drugs)

def run_model_comparison(drug_name, X_train, X_test, y_train, y_test, train_loader, 
                        input_dim, epochs=50, learning_rate=0.001):
    """
    Run model comparison for a specific drug.
    """
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON FOR: {drug_name}")
    print(f"{'='*60}")
    
    results = {}
    # Create a new classifier for this drug's feature dimension
    num_classes = 24
    input_dim = X_train.shape[1]
    clf = MLPClassifier(input_dim, num_classes)
    clf.load_state_dict(torch.load("models/mlp_classifier_v_01.0.pt", map_location=torch.device('cpu')))

    # Don't load pretrained weights since dimensions don't match
    # Instead, train from scratch or skip this model
    print(f"Creating new MLPClassifier for {input_dim} features (no pretrained weights)")
    
    # === 1. Linear Regression (Sklearn) ===
    print(f"\n=== 1. Linear Regression (Sklearn) ===")
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
    print(f"\n=== 2. Random Forest ===")
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
        'spearman': spearman_rf,
        'model': rf  # Store model for feature importance
    }
    
    # === 3. Simple Neural Network ===
    print(f"\n=== 3. Simple Neural Network ===")
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
    print(f"\n=== 4. MLP Regressor (Reset Weights) ===")

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
    
    return results

def plot_drug_results(drug_name, results, y_test, save_dir="plots"):
    """
    Create comprehensive plots for drug results and save them to folder.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
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
    for model_name, metrics in results.items():
        plot_residuals(torch.tensor(metrics['y_pred']), y_test, 
                       text=f"GDRP Residuals – {drug_name} – {model_name}", 
                       y_lim=y_lim, x_lim=x_lim)
        
        # Save residual plot
        safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        safe_drug_name = drug_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.savefig(f"{save_dir}/residuals_{safe_drug_name}_{safe_model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Comparison plot of predictions vs actual
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'orange', 'red', 'green']
    markers = ['o', 's', '^', 'D']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        plt.scatter(y_true_np, metrics['y_pred'], 
                   alpha=0.6, label=f"{model_name} (Pearson={metrics['pearson']:.3f})", 
                   color=colors[i], marker=markers[i])
    
    # Perfect prediction line
    min_val = min(y_true_np.min(), min(all_predictions))
    max_val = max(y_true_np.max(), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.xlabel('Actual AUC')
    plt.ylabel('Predicted AUC')
    plt.title(f'Model Comparison: {drug_name} - Predicted vs Actual AUC Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save comparison plot
    safe_drug_name = drug_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    plt.savefig(f"{save_dir}/comparison_{safe_drug_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/ directory")
    
    # Bar chart comparison of metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in models]
    r2_values = [results[model]['r2'] for model in models]
    pearson_values = [results[model]['pearson'] for model in models]
    spearman_values = [results[model]['spearman'] for model in models]
    
    # RMSE (lower is better)
    bars1 = ax1.bar(models, rmse_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
    ax1.set_title(f'RMSE Comparison - {drug_name} (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # R² (higher is better)
    bars2 = ax2.bar(models, r2_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
    ax2.set_title(f'R² Score Comparison - {drug_name} (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Pearson Correlation (higher is better)
    bars3 = ax3.bar(models, pearson_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
    ax3.set_title(f'Pearson Correlation - {drug_name} (Higher is Better)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Pearson Correlation')
    ax3.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars3, pearson_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Spearman Correlation (higher is better)
    bars4 = ax4.bar(models, spearman_values, color=['blue', 'orange', 'red', 'green'], alpha=0.7)
    ax4.set_title(f'Spearman Correlation - {drug_name} (Higher is Better)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Spearman Correlation')
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars4, spearman_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.close()
    
    # Feature importance for Random Forest
    if 'Random Forest' in results and 'model' in results['Random Forest']:
        rf_model = results['Random Forest']['model']
        feature_importance = rf_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
        plt.yticks(range(len(top_features_idx)), [f'Feature {i}' for i in top_features_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Most Important Features - {drug_name} (Random Forest)', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.close()

from src.utils import check_drug_availability

def main():
    """
    Main function to run experiments for specific drugs.
    """
    print("=== DRUG-SPECIFIC MODEL COMPARISON ===")
    
    # List of specific drugs to test
    drugs_to_test = [
        'KU-55933', 'Sorafenib', 'CHIR-99021', 'Axitinib', 'BMS-536924',
        'GSK1904529A', 'AICAR', 'FK866', 'AZ628', 'BMS-754807', 'JQ1',
        'MK-2206', 'PAC-1', 'GDC0941', 'PD-0332991', 'JNK_Inhibitor_VIII',
        'Nilotinib', 'AZD6482', 'AZD8055', 'BX-795', 'PF-562271', 'YK_4-279',
        'LY317615', 'ZM-447439', 'PLX4720', 'NU-7441', 'PD-0325901',
        'Embelin', 'Obatoclax_mesylate', 'SL_0101-1', 'PD-173074', 'AMG-706',
        'GSK269962A', 'AZD7762', 'AG-014699', 'Camptothecin', 'BI-2536',
        'JNJ-26854165', '5Z-7-Oxozeaenol', 'SB_216763', 'Bosutinib',
        'Gefitinib', 'GW_441756', 'PF-4708671', 'Vorinostat', 'Tamoxifen',
        'CEP-701', '681640', 'RO-3306', 'TW_37'
    ]
    
    # Check which drugs are available
    available_drugs = check_drug_availability(drugs_to_test)
    
    if not available_drugs:
        print("No drugs from the list are available in the dataset!")
        return
    
    print(f"\nProceeding with {len(available_drugs)} available drugs...")
    
    # Store all results
    all_results = {}
    
    for drug_name in available_drugs:
        try:
            # Load data for this drug
            X_train, X_test, y_train, y_test, train_loader = load_drug_data(drug_name)
            input_dim = X_train.shape[1]
            
            # Run model comparison
            results = run_model_comparison(drug_name, X_train, X_test, y_train, y_test, 
                                         train_loader, input_dim)
            
            # Store results
            all_results[drug_name] = results
            
            # Print summary for this drug
            print(f"\n{'='*60}")
            print(f"SUMMARY FOR {drug_name}")
            print(f"{'='*60}")
            print(f"{'Model':<25} {'RMSE':<10} {'R²':<10} {'Pearson':<10} {'Spearman':<10}")
            print("-"*60)
            for model_name, metrics in results.items():
                print(f"{model_name:<25} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f} "
                      f"{metrics['pearson']:<10.4f} {metrics['spearman']:<10.4f}")
            
            # Find best model for this drug
            best_r2 = max(results.items(), key=lambda x: x[1]['r2'])
            best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])
            best_pearson = max(results.items(), key=lambda x: x[1]['pearson'])
            best_spearman = max(results.items(), key=lambda x: x[1]['spearman'])
            
            print(f"\nBest models for {drug_name}:")
            print(f"  Best R²: {best_r2[0]} (R²={best_r2[1]['r2']:.4f})")
            print(f"  Best RMSE: {best_rmse[0]} (RMSE={best_rmse[1]['rmse']:.4f})")
            print(f"  Best Pearson: {best_pearson[0]} (Pearson={best_pearson[1]['pearson']:.4f})")
            print(f"  Best Spearman: {best_spearman[0]} (Spearman={best_spearman[1]['spearman']:.4f})")
            
            # Highlight the best model based on Pearson correlation
            print(f"\n🏆 BEST MODEL (by Pearson correlation): {best_pearson[0]} (Pearson={best_pearson[1]['pearson']:.4f})")
            
            # Create plots
            plot_drug_results(drug_name, results, y_test)
            
        except Exception as e:
            print(f"Error processing drug {drug_name}: {e}")
            continue
    
    # Final summary
    if all_results:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Successfully processed {len(all_results)} drugs:")
        for drug_name in all_results.keys():
            print(f"  - {drug_name}")
        
        # Calculate overall best models based on Pearson correlation
        print(f"\nOverall best models across all drugs (by Pearson correlation):")
        best_models = {}
        for drug_name, results in all_results.items():
            best_pearson = max(results.items(), key=lambda x: x[1]['pearson'])
            best_models[drug_name] = best_pearson[0]
            print(f"  {drug_name}: {best_pearson[0]} (Pearson={best_pearson[1]['pearson']:.4f})")
        
        # Count how many times each model is the best
        model_counts = {}
        for model_name in best_models.values():
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        print(f"\nModel performance summary (by Pearson correlation):")
        for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_results)) * 100
            print(f"  {model_name}: {count}/{len(all_results)} drugs ({percentage:.1f}%)")
    
    print("\nDrug-specific comparison completed!")

if __name__ == "__main__":
    main()

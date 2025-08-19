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

def check_data_file(data_path="data/merged_df_all_drugs.parquet"):
    """
    Check if the data file exists and is accessible.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please make sure the file exists and the path is correct.")
        return False
    return True

def load_drug_data(drug_name, data_path="data/merged_df_all_drugs.parquet"):
    """
    Load data for a specific drug from the merged dataframe.
    """
    print(f"Loading data for drug: {drug_name}")
    
    # Check if file exists
    if not check_data_file(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the full dataset
    df = pd.read_parquet(data_path)
    
    # Filter data for the specific drug
    drug_df = df[df['DRUG_NAME'] == drug_name].copy()
    
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

def get_available_drugs(data_path="data/merged_df_all_drugs.parquet"):
    """
    Get list of available drugs in the dataset.
    """
    # Check if file exists
    if not check_data_file(data_path):
        return []
    
    df = pd.read_parquet(data_path)
    drugs = df['DRUG_NAME'].unique()
    print(f"Available drugs: {len(drugs)}")
    for i, drug in enumerate(sorted(drugs)):
        count = len(df[df['DRUG_NAME'] == drug])
        print(f"  {i+1:2d}. {drug:<20} ({count:4d} samples)")
    return sorted(drugs)

def run_model_comparison(drug_name, X_train, X_test, y_train, y_test, train_loader, 
                        input_dim, epochs=30, learning_rate=0.001):
    """
    Run model comparison for a specific drug.
    """
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON FOR: {drug_name}")
    print(f"{'='*60}")
    
    results = {}
    
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
    
    # Create a new classifier for this drug's feature dimension
    num_classes = 24
    clf = MLPClassifier(input_dim, num_classes)
    clf.load_state_dict(torch.load("models/mlp_classifier_v_01.0.pt", map_location=torch.device('cpu')))

    # Don't load pretrained weights since dimensions don't match
    # Instead, train from scratch or skip this model
    print(f"Creating new MLPClassifier for {input_dim} features (no pretrained weights)")
    
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
    Create plots for drug results and save them to folder.
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
                   alpha=0.6, label=f"{model_name} (R²={metrics['r2']:.3f})", 
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

def plot_overall_comparison(model_averages, all_results, save_dir="plots"):
    """
    Create overall comparison plots across all drugs.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Bar chart of average R² scores
    plt.figure(figsize=(12, 8))
    models = list(model_averages.keys())
    avg_r2_scores = [model_averages[model]['avg_r2'] for model in models]
    std_r2_scores = [model_averages[model]['std_r2'] for model in models]
    
    bars = plt.bar(models, avg_r2_scores, yerr=std_r2_scores, capsize=5, 
                   alpha=0.7, color=['blue', 'orange', 'red', 'green'])
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Models')
    plt.ylabel('Average R² Score')
    plt.title('Average R² Scores Across All Drugs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/overall_r2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart of average RMSE scores
    plt.figure(figsize=(12, 8))
    avg_rmse_scores = [model_averages[model]['avg_rmse'] for model in models]
    
    bars = plt.bar(models, avg_rmse_scores, alpha=0.7, color=['blue', 'orange', 'red', 'green'])
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_rmse_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Models')
    plt.ylabel('Average RMSE')
    plt.title('Average RMSE Scores Across All Drugs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/overall_rmse_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of R² scores for each drug and model
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    drugs = list(all_results.keys())
    model_names = ['Linear Regression', 'Random Forest', 'Simple Neural Network', 'MLP Regressor (Reset)']
    
    heatmap_data = []
    for model in model_names:
        row = []
        for drug in drugs:
            if model in all_results[drug]:
                row.append(all_results[drug][model]['r2'])
            else:
                row.append(0)  # or np.nan
        heatmap_data.append(row)
    
    # Create heatmap
    im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('R² Score')
    
    # Set labels
    plt.xticks(range(len(drugs)), drugs, rotation=45, ha='right')
    plt.yticks(range(len(model_names)), model_names)
    plt.xlabel('Drugs')
    plt.ylabel('Models')
    plt.title('R² Scores Heatmap: Models vs Drugs')
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(drugs)):
            if heatmap_data[i][j] != 0:
                plt.text(j, i, f'{heatmap_data[i][j]:.3f}', 
                        ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/r2_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot of R² scores distribution
    plt.figure(figsize=(12, 8))
    
    box_data = []
    for model in model_names:
        scores = []
        for drug in drugs:
            if model in all_results[drug]:
                scores.append(all_results[drug][model]['r2'])
        box_data.append(scores)
    
    plt.boxplot(box_data, labels=model_names, patch_artist=True)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Distribution of R² Scores Across Drugs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/r2_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overall comparison plots saved to {save_dir}/ directory")

def main():
    """
    Main function to run experiments for multiple drugs.
    """
    print("=== MULTI-DRUG MODEL COMPARISON ===")
    
    # Get available drugs
    available_drugs = get_available_drugs()
    
    if not available_drugs:
        print("No drugs found or data file is not accessible.")
        print("Please check if the file 'data/merged_df_all_drugs.parquet' exists and contains data.")
        return
    
    # Select drugs to test (you can modify this list)
    drugs_to_test = available_drugs[:5]  # Test first 5 drugs
    print(f"\nTesting drugs: {drugs_to_test}")
    
    # Store all results
    all_results = {}
    
    for drug_name in drugs_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {drug_name}")
            print(f"{'='*60}")
            
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
            print(f"\nBest model for {drug_name}: {best_r2[0]} (R²={best_r2[1]['r2']:.4f})")
            
            # Create plots
            plot_drug_results(drug_name, results, y_test)
            
        except Exception as e:
            print(f"Error processing drug {drug_name}: {e}")
            print(f"Skipping {drug_name} and continuing with next drug...")
            continue
    
    # Check if any drugs were successfully processed
    if not all_results:
        print("\nNo drugs were successfully processed. Please check the data file and drug names.")
        print("Possible issues:")
        print("1. Data file doesn't exist or is corrupted")
        print("2. Drug names don't match those in the dataset")
        print("3. Data format is incorrect")
        return
    
    print(f"\nSuccessfully processed {len(all_results)} drugs: {list(all_results.keys())}")
    
    # Create overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON ACROSS ALL DRUGS")
    print(f"{'='*80}")
    
    # Calculate average performance for each model
    model_averages = {}
    for model_name in ['Linear Regression', 'Random Forest', 'Simple Neural Network', 'MLP Regressor (Reset)']:
        r2_scores = []
        rmse_scores = []
        pearson_scores = []
        spearman_scores = []
        
        for drug_name, drug_results in all_results.items():
            if model_name in drug_results:
                r2_scores.append(drug_results[model_name]['r2'])
                rmse_scores.append(drug_results[model_name]['rmse'])
                pearson_scores.append(drug_results[model_name]['pearson'])
                spearman_scores.append(drug_results[model_name]['spearman'])
        
        if r2_scores:
            model_averages[model_name] = {
                'avg_r2': np.mean(r2_scores),
                'avg_rmse': np.mean(rmse_scores),
                'avg_pearson': np.mean(pearson_scores),
                'avg_spearman': np.mean(spearman_scores),
                'std_r2': np.std(r2_scores),
                'count': len(r2_scores)
            }
    
    # Check if we have any model averages
    if not model_averages:
        print("No model averages could be calculated. Please check the results.")
        return
    
    # Print overall summary
    print(f"{'Model':<25} {'Avg R²':<10} {'Std R²':<10} {'Avg RMSE':<10} {'Count':<10}")
    print("-"*80)
    for model_name, stats in model_averages.items():
        print(f"{model_name:<25} {stats['avg_r2']:<10.4f} {stats['std_r2']:<10.4f} "
              f"{stats['avg_rmse']:<10.4f} {stats['count']:<10}")
    
    # Find overall best model
    best_overall = max(model_averages.items(), key=lambda x: x[1]['avg_r2'])
    print(f"\nOverall best model: {best_overall[0]} (Avg R²={best_overall[1]['avg_r2']:.4f})")
    
    # Create overall comparison plots
    plot_overall_comparison(model_averages, all_results, save_dir="plots")
    
    print("\nMulti-drug comparison completed!")

if __name__ == "__main__":
    main()

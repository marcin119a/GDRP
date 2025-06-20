import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from src.utils import set_seed
import torch.nn as nn
from src.models import MLPClassifier, MLPRegressor
from src.train_classifier import train_model

set_seed(42)

# === Load and prepare the data ===
df = pd.read_parquet("data/combined_tcga_df.parquet")

# Keep only the top 24 most common cancer types
top_sites = df["primary_site"].value_counts().nlargest(24).index
filtered_df = df[df["primary_site"].isin(top_sites)]

# Encode target labels as integers
label_encoder = LabelEncoder()
filtered_df["primary_site_encoded"] = label_encoder.fit_transform(filtered_df["primary_site"])

# Prepare features and labels
X = np.array(filtered_df["feature_vector"].tolist())
y = np.array(filtered_df["primary_site_encoded"])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# === Training function ===

# === Initialize and train model ===
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = MLPClassifier(input_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, optimizer, num_epochs=50)

# === Evaluate the model ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1)
    precision = precision_score(y_test.numpy(), y_pred.numpy(), average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# === Save model to disk ===
torch.save(model.state_dict(), "models/mlp_classifier_v_02.0.pt")
print("Model saved as models/mlp_classifier_v_02.0.pt")
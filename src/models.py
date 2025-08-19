import torch.nn as nn
import torch


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)


class MLPRegressor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(pretrained_model.model.children())[:-1])
        self.regression_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.loss_fn = nn.MSELoss()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regression_head(features)


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        return self.linear(x)


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)


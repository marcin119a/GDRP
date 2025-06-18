import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )

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
import numpy as np
from src.metrics import pearson_corr

def train(model, dataloader, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        losses, p_corrs = [], []
        for Xb, yb in dataloader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = model.loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            p_corrs.append(pearson_corr(out.squeeze(), yb.squeeze()).item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}, Pearson: {np.mean(p_corrs):.4f}")

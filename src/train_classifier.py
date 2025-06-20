

def train_model(model, dataloader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct, total = 0, 0
        for Xb, yb in dataloader:
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = model.loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

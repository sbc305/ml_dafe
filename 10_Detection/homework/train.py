import wandb
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score

from model import create_model, load_model
from load_data import get_loader

model = create_model()
train_loader, val_loader = get_loader()

wandb.init(project="face-recognition")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def evaluate(model, loader, save=True, save_path="./checkpoints"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    return acc, prec

for epoch in range(10):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc, val_prec = evaluate(model, val_loader)
    wandb.log({
        "Epoch": epoch + 1,
        "Loss": running_loss / len(train_loader),
        "Val Accuracy": val_acc,
        "Val Precision": val_prec
    })
    if save:
        save_path = f"./checkpoints/model_{epoch}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        wandb.save(save_path)
        print(f"Model saved to {save_path}")

print("Training complete.")
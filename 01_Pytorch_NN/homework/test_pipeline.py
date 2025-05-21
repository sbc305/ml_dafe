import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange
import wandb


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result

@pytest.fixture
def train_dataset():
    # note: реализуйте и протестируйте подготовку данных (скачиание и препроцессинг)

    train_dataset = CIFAR10("CIFAR10/train", download=True)
    train_dataset = CIFAR10(root="CIFAR10/train", train=True, transform=transform, download=False)

    return train_dataset

@pytest.fixture
def test_dataset():
    test_dataset = CIFAR10("CIFAR10/test", download=True)
    test_dataset = CIFAR10(root="CIFAR10/test", train=False, transform=transform, download=False)

    return test_dataset

configs = [
    dict(device="cpu", batch_size=64, learning_rate=1e-4, weight_decay=0.1, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=64, learning_rate=1e-4, weight_decay=0.05, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=64, learning_rate=1e-5, weight_decay=0.1, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=64, learning_rate=1e-5, weight_decay=0.05, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=32, learning_rate=1e-4, weight_decay=0.1, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=32, learning_rate=1e-4, weight_decay=0.05, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=32, learning_rate=1e-5, weight_decay=0.1, epochs=2, zero_init_residual=False),
    dict(device="cpu", batch_size=32, learning_rate=1e-5, weight_decay=0.05, epochs=2, zero_init_residual=False),
    ]

@pytest.mark.parametrize("config", configs)
def test_train_on_one_batch(config, train_dataset, test_dataset):
    # note: реализуйте и протестируйте один шаг обучения вместе с метрикой
    model = resnet18(weights=None, num_classes=10, zero_init_residual=config["zero_init_residual"])
    device = config["device"]
    model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config["batch_size"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert loss.item() > 0

configs = [
    dict(device="cpu", batch_size=64, learning_rate=1e-4, weight_decay=0.1, epochs=2, zero_init_residual=False),
    ]
@pytest.mark.parametrize("config", configs)
def test_training(config, train_dataset, test_dataset):
    # note: реализуйте и протестируйте полный цикл обучения модели (обучение, валидацию, логирование, сохранение артефактов)
    
    wandb.init(config=config, project="effdl_example", name="baseline")
    model = resnet18(weights=None, num_classes=10, zero_init_residual=config["zero_init_residual"])
    device = config["device"]
    model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config["batch_size"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    for epoch in trange(config["epochs"]):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                all_preds = []
                all_labels = []

                for test_images, test_labels in test_loader:
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    with torch.inference_mode():
                        outputs = model(test_images)
                        preds = torch.argmax(outputs, 1)

                        all_preds.append(preds)
                        all_labels.append(test_labels)

                accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))

                metrics = {'test_acc': accuracy, 'train_loss': loss}
                wandb.log(metrics, step=epoch * len(train_dataset) + (i + 1) * config["batch_size"])
    torch.save(model.state_dict(), "model.pt")

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on FER-2013')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='path to data directory (should contain train/ and test/)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping patience (epochs without improvement)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_dir if args.data_dir else os.path.join(project_root, 'data')
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # Hyperparameters
    num_classes = 7
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    patience = args.patience

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model setup
    resnet_weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=resnet_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop with progress bar
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        # Batch-level progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_loss /= len(test_loader.dataset)
        val_acc = 100 * correct / total

        # Scheduler step
        scheduler.step(val_loss)

        # End-of-epoch logging
        print(f"Epoch {epoch}/{num_epochs} — "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            save_dir = os.path.join(project_root, 'models')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_resnet18.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training complete.")

if __name__ == '__main__':
    main()

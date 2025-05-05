import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on FER-2013 with advanced fine-tuning')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='path to data directory (should contain train/ and test/)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience (epochs without improvement)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay for optimizer')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_dir if args.data_dir else os.path.join(project_root, 'data')
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    num_classes = 7
    batch_size = args.batch_size
    num_epochs = args.epochs
    initial_lr = args.lr
    patience = args.patience

    # Data transforms with refined augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Compute class weights with epsilon and normalize
    targets = [label for _, label in train_dataset.samples]
    class_counts = [0] * num_classes
    for label in targets:
        class_counts[label] += 1
    class_weights = []
    for count in class_counts:
        class_weights.append(1.0 / (count + 1e-6))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float)
    weight_tensor = weight_tensor / weight_tensor.sum() * num_classes

    sampler = WeightedRandomSampler(
        [class_weights[label] for label in targets],
        num_samples=len(targets), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    # Model setup: unfreeze more layers
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for name, param in model.named_parameters():
        if not (name.startswith('layer2') or name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    # Loss with softer label smoothing
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device), label_smoothing=0.05)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=initial_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.5,
        div_factor=10,
        final_div_factor=100
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation with NaN checks
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print("Warning: NaN loss in validation batch, skipping.")
                    continue
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_loss /= max(len(test_loader.dataset), 1)
        val_acc = 100 * correct / max(total, 1)

        print(f"Epoch {epoch}/{num_epochs} — Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(project_root, 'models', 'best_resnet18.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training complete.")

if __name__ == '__main__':
    main()

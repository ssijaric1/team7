# === main2_vscode.py ===
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt

def mixup_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 on FER-2013')
    parser.add_argument('--data-dir', type=str, default='data', help='path to train/ and test/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--mixup-alpha', type=float, default=0.4)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir  = os.path.join(args.data_dir, 'test')

    # Data transforms (reduced augmentation)
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_tf)

    targets = [lbl for _, lbl in train_ds.samples]
    counts = Counter(targets)
    num_cls = len(train_ds.classes)
    class_weights = torch.tensor([1.0/(counts[i]+1e-6) for i in range(num_cls)], device=device)
    class_weights = class_weights / class_weights.sum() * num_cls

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_f, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(256, num_cls)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_loss = float('inf')
    no_improve = 0
    train_accs, val_accs = [], []

    for ep in range(1, args.epochs+1):
        model.train()
        correct = 0; total = 0; train_loss = 0
        for imgs, lbls in tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs, y_a, y_b, lam = mixup_data(imgs, lbls, args.mixup_alpha, device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        train_loss /= len(train_ld.dataset)
        train_acc = correct / total
        train_accs.append(train_acc)

        # Validation
        model.eval(); val_loss = 0; correct = 0; total = 0
        preds_list = []; targets_list = []
        with torch.no_grad():
            for imgs, lbls in test_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

                preds_list += preds.cpu().tolist()
                targets_list += lbls.cpu().tolist()

        val_loss /= len(test_ld.dataset)
        val_acc = correct / total
        val_accs.append(val_acc)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(classification_report(targets_list, preds_list, digits=4))
        print(confusion_matrix(targets_list, preds_list))

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best.pth')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping.")
                break

    # === Plotting training vs validation accuracy ===
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('acc_plot.png')
    plt.show()

if __name__ == '__main__':
    main()

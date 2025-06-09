# === main2_vscode.py ===
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# MixUp utility
def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda''' 
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
    parser = argparse.ArgumentParser(description='Train ResNet on FER-2013 (local)')
    parser.add_argument('--data-dir', type=str, default='data', help='path to train/ and test/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--mixup-alpha', type=float, default=0.4, help='MixUp alpha')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')

    # Augmentations
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.Lambda(lambda img: img + 0.05*torch.randn_like(img)),  # Gaussian noise
        transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_tf)

    # Class weights
    targets = [lbl for _, lbl in train_ds.samples]
    counts  = Counter(targets)
    num_cls = len(train_ds.classes)
    class_weights = torch.tensor([1.0/(counts[i]+1e-6) for i in range(num_cls)], device=device)
    class_weights = class_weights / class_weights.sum() * num_cls

    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size,
                          shuffle=False, num_workers=4)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for n,p in model.named_parameters():
        if not (n.startswith('layer1') or n.startswith('layer2')
                or n.startswith('layer3') or n.startswith('layer4')
                or n.startswith('fc')):
            p.requires_grad=False
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f,256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256,num_cls)
    )
    model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_loss = float('inf')
    no_improve     = 0

    for ep in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            # MixUp
            imgs, y_a, y_b, lam = mixup_data(imgs, lbls, args.mixup_alpha, device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = lam * criterion(outputs, y_a) + (1-lam) * criterion(outputs, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_loss += loss.item()*imgs.size(0)
        train_loss /= len(train_ld.dataset)

        # Validation
        model.eval(); val_loss=0; preds_list=[]; targets_list=[]
        with torch.no_grad():
            for imgs, lbls in test_ld:
                imgs = imgs.to(device); lbls = lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item()*imgs.size(0)
                preds_list += out.argmax(1).cpu().tolist()
                targets_list += lbls.cpu().tolist()
        val_loss /= len(test_ld.dataset)
        print(f"Val Loss: {val_loss:.4f}")

        # Metrics
        print(classification_report(targets_list, preds_list, digits=4))
        print(confusion_matrix(targets_list, preds_list))

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; no_improve=0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best.pth')
        else:
            no_improve+=1
            if no_improve>=args.patience:
                print("Early stopping.")
                break

if __name__ == '__main__':
    main()
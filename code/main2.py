import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ResNet on FER-2013 locally (VS Code)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='path to data directory (train/ and test/ inside)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir  = os.path.join(args.data_dir, 'test')

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    # datasets & sampler
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=test_tf)
    targets  = [lbl for _, lbl in train_ds.samples]
    counts   = Counter(targets)
    num_cls  = len(train_ds.classes)
    cw       = [1.0/(counts[i]+1e-6) for i in range(num_cls)]
    # normalize
    w_tensor = torch.tensor(cw)
    w_tensor = w_tensor / w_tensor.sum() * num_cls
    sampler  = WeightedRandomSampler([cw[l] for l in targets],
                                     num_samples=len(targets), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=2)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size,
                          shuffle=False,   num_workers=2)

    # model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for n,p in model.named_parameters():
        if not (n.startswith('layer2') or n.startswith('layer3')
                or n.startswith('layer4') or n.startswith('fc')):
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
    model = model.to(device)

    # loss / optim / sched
    crit = nn.CrossEntropyLoss(weight=w_tensor.to(device),
                               label_smoothing=0.05)
    opt  = optim.Adam([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=args.weight_decay)
    sched= optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        steps_per_epoch=len(train_ld),
        epochs=args.epochs,
        pct_start=0.5,
        div_factor=10,
        final_div_factor=100
    )

    best_loss = float('inf')
    no_imp    = 0

    for ep in range(1, args.epochs+1):
        model.train()
        run_loss = 0.0
        for imgs, lbls in tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss= crit(out,lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()*imgs.size(0)
        train_loss = run_loss/len(train_ld.dataset)

        # validation
        model.eval()
        v_loss=0.0; corr=0; tot=0
        with torch.no_grad():
            for imgs, lbls in test_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss= crit(out,lbls)
                if torch.isnan(loss): continue
                v_loss += loss.item()*imgs.size(0)
                preds = out.argmax(1)
                corr += (preds==lbls).sum().item()
                tot  += lbls.size(0)
        val_loss = v_loss/ max(tot,1)
        val_acc  = 100*corr/ max(tot,1)
        print(f"Epoch {ep}/{args.epochs} — "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
              f"Acc: {val_acc:.2f}%")

        if val_loss<best_loss:
            best_loss=v.val_loss
            no_imp=0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(),'models/best.pth')
        else:
            no_imp+=1
            if no_imp>=args.patience:
                print("Early stopping.")
                break

if __name__=='__main__':
    main()
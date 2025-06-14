import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets.folder import default_loader


def mixup_data(x, y, alpha=1.0, device='cpu'):

    # mixup augmentation:
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def parse_args():

    # parse training arguments:
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 on FER-2013')
    parser.add_argument('--data-dir', type=str, default='data', help='path to train/ and test/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--mixup-alpha', type=float, default=0.4)
    args, _ = parser.parse_known_args()
    return args


def test_emotion_distribution(emotion, model, device, transform, dataset, num_samples=50):

    # testing distribution for one emotion:
    classes = dataset.classes
    if emotion not in classes:
        raise ValueError(f"Emotion '{emotion}' not found in classes: {classes}")
    idx = classes.index(emotion)
    probs = []
    count = 0
    model.eval()
    with torch.no_grad():
        for path, label in dataset.samples:
            if label != idx:
                continue
            img = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
            outputs = model(img)
            probs.append(torch.softmax(outputs, dim=1)[0, idx].item())
            count += 1
            if count >= num_samples:
                break

    plt.figure(figsize=(8,4))
    plt.hist(probs, bins=10)
    plt.xlabel(f'Predicted Probability for {emotion}')
    plt.ylabel('Frequency')
    plt.title(f'Confidence for {emotion} (n={len(probs)})')
    plt.grid(True)
    plt.savefig(f'{emotion.lower()}_dist.png')
    plt.show()


class FilteredImageDataset(Dataset):
    
    # filtering classes that are not present in training set:
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        self.classes = list(class_to_idx.keys())
        
        # build a list of valid samples (so only classes in training set):
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in class_to_idx:
                continue
                
            class_idx = class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')

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

    # creating the training dataset:
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    num_cls = len(train_ds.classes)
    
    # creating the filtered test dataset using training classes:
    test_ds = FilteredImageDataset(
        test_dir,
        class_to_idx=train_ds.class_to_idx,
        transform=test_tf
    )
    test_ds.classes = train_ds.classes
    test_ds.class_to_idx = train_ds.class_to_idx

    print(f"Training classes ({num_cls}): {train_ds.classes}")
    print(f"Test classes after filtering: {test_ds.classes}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    targets = [lbl for _, lbl in train_ds.samples]
    counts = Counter(targets)
    class_weights = torch.tensor([1.0/(counts[i]+1e-6) for i in range(num_cls)], device=device)
    class_weights = class_weights / class_weights.sum() * num_cls

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # setting up the model:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
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
    
    # optimizer, loss, and scheduler:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.lr, 
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val = float('inf')
    no_imp = 0
    train_accs, val_accs = [], []

    # training:
    for ep in range(1, args.epochs+1):
        model.train()
        total, correct, train_loss = 0, 0, 0
        for imgs, lbls in tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs, y_a, y_b, lam = mixup_data(imgs, lbls, args.mixup_alpha, device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = lam*criterion(out, y_a)+(1-lam)*criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_loss += loss.item()*imgs.size(0)
            preds = out.argmax(1)
            correct += (preds==lbls).sum().item()
            total += lbls.size(0)
        train_accs.append(correct/total)

        # validation:
        model.eval()
        total, correct, val_loss = 0,0,0
        preds_l, targets_l = [],[]
        with torch.no_grad():
            for imgs, lbls in test_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_loss += criterion(out,lbls).item()*imgs.size(0)
                preds = out.argmax(1)
                correct += (preds==lbls).sum().item()
                total += lbls.size(0)
                preds_l += preds.cpu().tolist()
                targets_l += lbls.cpu().tolist()
        val_accs.append(correct/total)
        print(f"Val Loss: {val_loss/total:.4f} Acc: {correct/total:.4f}")
        print(classification_report(targets_l,preds_l,digits=4))
        scheduler.step(val_loss)
        if val_loss<best_val:
            best_val=val_loss
            no_imp=0
            os.makedirs('models',exist_ok=True)
            torch.save(model.state_dict(),'models/best.pth')
        else:
            no_imp+=1
            if no_imp>=args.patience:
                print("Early stopping")
                break


    plt.figure(figsize=(10,5))
    plt.plot(train_accs,label='Train')
    plt.plot(val_accs,label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()
    plt.savefig('acc_plot.png')
    plt.show()


    model.load_state_dict(torch.load('models/best.pth'))
    test_emotion_distribution('Happy', model, device, test_tf, test_ds)

if __name__=='__main__':
    main()
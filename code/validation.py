import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
from main import FilteredImageDataset, test_emotion_distribution

def parse_args():
    p = argparse.ArgumentParser(description="Validate EfficientNet-B0 on FER-2013")
    p.add_argument('--data-dir',     type=str, default='data', help='path to train/ and test/')
    p.add_argument('--batch-size',   type=int, default=32)
    p.add_argument('--model-path',   type=str, default='models/best.pth')
    p.add_argument('--emotion',      type=str, default=None,
                   help="If set, plot the confidence distribution for this class")
    p.add_argument('--num-samples',  type=int, default=50,
                   help="How many examples of that emotion to sample")
    return p.parse_args()

def load_test_tf_and_dataset(data_dir):
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=test_tf)
    test_ds  = FilteredImageDataset(
        root_dir=os.path.join(data_dir, 'test'),
        class_to_idx=train_ds.class_to_idx,
        transform=test_tf
    )
    test_ds.classes      = train_ds.classes
    test_ds.class_to_idx = train_ds.class_to_idx

    return test_tf, test_ds

def build_model(num_classes, device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_f = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.6),
        torch.nn.Linear(in_f, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(256, num_classes)
    )
    return model.to(device)

def validate_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    test_tf, test_ds = load_test_tf_and_dataset(args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Loaded {len(test_ds)} test images across {len(test_ds.classes)} classes.\n")

    model = build_model(num_classes=len(test_ds.classes), device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded weights from '{args.model_path}'\n")

    validate_all(model, test_loader, device)

    if args.emotion:
        print(f"\n→ Plotting confidence distribution for '{args.emotion}' ({args.num_samples} samples)")
        test_emotion_distribution(
            emotion=args.emotion,
            model=model,
            device=device,
            transform=test_tf,
            dataset=test_ds,
            num_samples=args.num_samples
        )
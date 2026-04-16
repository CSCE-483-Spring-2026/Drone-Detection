from collections import Counter
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch.nn.functional as F
from data_loader import load_hf_non_drone_dataset, load_local_drone_audio_dataset, aggregate_datasets, train_valid_split, DroneAudioDataset
import numpy as np

class CNNModel(torch.nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)

        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = torch.nn.Dropout(0.5)

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def single_epoch(model, data_loader, criterion, optimizer, mean=None, std=None, SPEC_H=None, SPEC_W=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    device = next(model.parameters()).device

    for batch in data_loader:
        x_batch = batch["x"].float().to(device)
        y_batch = batch["y"].long().to(device)

        x_batch = (x_batch - mean) / (std + 1e-8)

        x_batch = x_batch.view(x_batch.size(0), 1, SPEC_H, SPEC_W)

        optimizer.zero_grad()
        logits = model(x_batch) 
        loss = criterion(logits, y_batch) 
        loss.backward()
        optimizer.step() 

        total_loss += loss.item() * x_batch.size(0)

        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_correct/total, total_loss/total

@torch.no_grad()
def evaluate(model, data_loader, criterion, mean=None, std=None, SPEC_H=None, SPEC_W=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    device = next(model.parameters()).device
    all_predictions = []
    all_labels = []

    for batch in data_loader:
        x_batch = batch["x"].float().to(device)
        x_batch = (x_batch - mean) / (std + 1e-8)
        y_batch = batch["y"].long().to(device)

        x_batch = x_batch.view(x_batch.size(0), 1, SPEC_H, SPEC_W)

        logits = model(x_batch) 
        loss = criterion(logits, y_batch) 
        total_loss += loss.item() * x_batch.size(0)

        predictions = torch.argmax(logits, dim=1)

        total_correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    print("Predicted counts:", Counter(all_predictions))
    print("True counts:", Counter(all_labels))
    accuracy = total_correct / total
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])
    per_class_precision = precision_score(all_labels, all_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_recall = recall_score(all_labels, all_predictions, average=None, labels=[0, 1, 2], zero_division=0)   
    per_class_f1 = f1_score(all_labels, all_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    return accuracy, precision, recall, f1, total_loss/total, cm, per_class_precision, per_class_recall, per_class_f1

    

@torch.no_grad()
def compute_feature_mean_std(data_loader, device=None):
    sum_x = None
    sum_x2 = None
    n = 0
    for batch in data_loader:
        x_batch = batch["x"].float().to(device)
        if sum_x is None:
            sum_x = x_batch.sum(dim=0)
            sum_x2 = (x_batch**2).sum(dim=0)
        else:
            sum_x += x_batch.sum(dim=0)
            sum_x2 += (x_batch**2).sum(dim=0)
        n += x_batch.size(0)

    mean = sum_x / n
    var = (sum_x2 / n) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean, std


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = True if torch.cuda.is_available() else False
    print(f"Using device: {device}")

    hf_ds = load_hf_non_drone_dataset()
    local_ds = load_local_drone_audio_dataset()
    full_ds = aggregate_datasets(hf_ds, local_ds)

    train_ds, valid_ds = train_valid_split(full_ds)
    train_ds = train_ds.shuffle(seed=42)
    valid_ds = valid_ds.shuffle(seed=42)

    train_loader = DataLoader(DroneAudioDataset(train_ds,train=True, cap_length=10), batch_size=64, pin_memory=True)
    valid_loader = DataLoader(DroneAudioDataset(valid_ds, train=False, cap_length=10), batch_size=64, pin_memory=True)

    train_loader_for_mean_std = DataLoader(DroneAudioDataset(train_ds,train=False, cap_length=10), batch_size=64, pin_memory=True)
    mean, std = compute_feature_mean_std(train_loader_for_mean_std, device=device)

    print(f"mean/std shape: {mean.shape}, {std.shape}")
    print("mean stats:", mean.min().item(), mean.max().item(), mean.mean().item(), mean.std().item())
    print("std stats:", std.min().item(), std.max().item(), std.mean().item(), std.std().item())

    first_batch = next(iter(train_loader))
    print("first batch class counts:", Counter(first_batch["y"].tolist()))
    
    # need to verify that the input dimension matches the expected dimension for the CNN model
    SPEC_H = 64
    SPEC_W = first_batch['x'].shape[1] // SPEC_H
    model = CNNModel(input_channels=1, num_classes=3).to(device)

    def compute_window_class_counts(data_loader):
        counts = Counter()
        for batch in data_loader:
            y = batch["y"]
            for label in y.tolist():
                counts[label] += 1
        print("Window-level class counts:", counts)
        return counts
    
    count_loader = DataLoader(DroneAudioDataset(train_ds,train=False, cap_length=10), batch_size=64, pin_memory=True)

    counts = compute_window_class_counts(count_loader)
    class_counts = torch.tensor([counts[0], counts[1], counts[2]], dtype=torch.float32, device=device)

    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    print("Class weights:", class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 3
    best_accuracy = 0.0
    best_valid_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        train_acc, train_loss = single_epoch(model, train_loader, criterion, optimizer, mean=mean, std=std, SPEC_H=SPEC_H, SPEC_W=SPEC_W)
        valid_acc, valid_precision, valid_recall, valid_f1, valid_loss, confusion_mat, per_class_precision, per_class_recall, per_class_f1 = evaluate(model, valid_loader, criterion, mean=mean, std=std, SPEC_H=SPEC_H, SPEC_W=SPEC_W)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}, Precision: {valid_precision}, Recall: {valid_recall}, F1 Score: {valid_f1}")
        print(f"Per-class Precision: {per_class_precision}, Per-class Recall: {per_class_recall}, Per-class F1: {per_class_f1}")
        print(f"Confusion Matrix:\n{confusion_mat}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_accuracy = valid_acc
            best_precision = valid_precision
            best_recall = valid_recall
            best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_channels": 1,
                "num_classes": 3,
                "spec_h": SPEC_H,
                "spec_w": SPEC_W,
                "feat_mean": mean.detach().cpu(),
                "feat_std": std.detach().cpu(),
                "epoch" : epoch + 1,
                "accuracy": valid_acc,
                "precision": valid_precision,
                "recall": valid_recall,
                "f1_score": valid_f1
            }, "best_cnn_model.pth")

    # print small drone, large drone, non-drone window counts in train and valid sets
    print(f"Train set - Non-drone windows: {train_loader.dataset.non_drone_windows}, Small drone windows: {train_loader.dataset.small_drone_windows}, Large drone windows: {train_loader.dataset.large_drone_windows}")
    print(f"Validation set - Non-drone windows: {valid_loader.dataset.non_drone_windows}, Small drone windows: {valid_loader.dataset.small_drone_windows}, Large drone windows: {valid_loader.dataset.large_drone_windows}")

    print(f"Best Validation Accuracy: {best_accuracy:.4f}, F1 Score: {best_valid_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f} from epoch {best_epoch}")


if __name__ == "__main__":
    main()
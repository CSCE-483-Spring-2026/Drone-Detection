import torch
from torch.utils.data import DataLoader
from data_loader import load_drone_audio_dataset, train_valid_split, DroneAudioDataset
import numpy as np

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)  # output is a single logit for binary classification

    def forward(self, x):
        return self.linear(x) 
    
def pos_weight(train_loader):
    positive_count = 0
    negative_count = 0

    for batch in train_loader:
        y_batch = batch["y"].float().unsqueeze(1)

        positive_count += (y_batch == 1).sum().item()
        negative_count += (y_batch == 0).sum().item()
    
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)

def single_epoch(model, data_loader, criterion, optimizer, device=None, mean=None, std=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0

    for batch in data_loader:
        x_batch = batch["x"].float().to(next(model.parameters()).device)
        x_batch = (x_batch - mean) / (std + 1e-8)
        y_batch = batch["y"].float().unsqueeze(1).to(next(model.parameters()).device)

        optimizer.zero_grad()
        logits = model(x_batch) 
        loss = criterion(logits, y_batch) 
        loss.backward()
        optimizer.step() 

        total_loss += loss.item() * x_batch.size(0)

        with torch.no_grad():
            outputs = torch.sigmoid(logits)
            predictions = (outputs >= 0.5).float()
            total_correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_correct/total, total_loss/total

@torch.no_grad()
def evaluate(model, data_loader, criterion, device=None, mean=None, std=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for batch in data_loader:
        x_batch = batch["x"].float().to(next(model.parameters()).device)
        x_batch = (x_batch - mean) / (std + 1e-8)
        y_batch = batch["y"].float().unsqueeze(1).to(next(model.parameters()).device)

        logits = model(x_batch) 
        loss = criterion(logits, y_batch) 
        total_loss += loss.item() * x_batch.size(0)

        outputs = torch.sigmoid(logits)
        predictions = (outputs >= 0.5).float()

        total_correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

        true_positives += ((predictions == 1) & (y_batch == 1)).sum().item()
        false_positives += ((predictions == 1) & (y_batch == 0)).sum().item()
        true_negatives += ((predictions == 0) & (y_batch == 0)).sum().item()
        false_negatives += ((predictions == 0) & (y_batch == 1)).sum().item()
    
    accuracy = total_correct / total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score, total_loss/total, (true_positives, false_positives, true_negatives, false_negatives)

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
    print(f"Using device: {device}")

    dataset = load_drone_audio_dataset()
    
    train_ds, valid_ds = train_valid_split(dataset)

    train_loader = DataLoader(DroneAudioDataset(train_ds, cap_length=10,train=True), batch_size=1024, pin_memory=True)
    valid_loader = DataLoader(DroneAudioDataset(valid_ds, cap_length=10, train=False), batch_size=1024, pin_memory=True)

    train_loader_for_stats = DataLoader(DroneAudioDataset(train_ds, cap_length=10, train=False), batch_size=1024, pin_memory=True)
    mean, std = compute_feature_mean_std(train_loader_for_stats, device=device)

    print("mean/std shapes:", mean.shape, std.shape)
    print("mean stats:", mean.min().item(), mean.max().item(), mean.mean().item(), mean.std().item())
    print("std stats:", std.min().item(), std.max().item(), std.mean().item(), std.std().item())

    first_batch = next(iter(train_loader))
    input_size = first_batch['x'].shape[1]  # number of features
    model = LogisticRegressionModel(input_dim = input_size).to(device)

    print(next(model.parameters()).device)

    pw = pos_weight(train_loader).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 20
    best_accuracy = 0.0
    best_valid_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        train_acc, train_loss = single_epoch(model, train_loader, criterion, optimizer, device=device, mean=mean, std=std)
        valid_acc, valid_precision, valid_recall, valid_f1, valid_loss, (true_positives, false_positives, true_negatives, false_negatives) = evaluate(model, valid_loader, criterion, device=device, mean=mean, std=std)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1 Score: {valid_f1:.4f}")
        print(f"Confusion Matrix: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")

        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            best_valid_f1 = valid_f1
            best_precision = valid_precision
            best_recall = valid_recall
            best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_size,
                "feat_mean": mean.detach().cpu(),
                "feat_std": std.detach().cpu(),
            }, "best_logistic_model.pth")

    print(f"Train windows: {train_loader.dataset.drone_windows} drone, {train_loader.dataset.non_drone_windows} non-drone")
    print(f"Valid windows: {valid_loader.dataset.drone_windows} drone, {valid_loader.dataset.non_drone_windows} non-drone")

    print(f"Best Validation Accuracy: {best_accuracy:.4f}, F1 Score: {best_valid_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f} from epoch {best_epoch}")

if __name__ == "__main__":
    main()

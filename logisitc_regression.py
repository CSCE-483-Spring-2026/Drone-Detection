import torch
from torch.utils.data import DataLoader
from data_loader import load_drone_audio_dataset, train_valid_split, DroneAudioDataset

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

    



def single_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0

    for batch in data_loader:
        x_batch = batch["x"].float()
        y_batch = batch["y"].float().unsqueeze(1)

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
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for batch in data_loader:
        x_batch = batch["x"].float()
        y_batch = batch["y"].float().unsqueeze(1)

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
    


def main(cap_length=3, keep_prob=0.3):
    dataset = load_drone_audio_dataset()
    
    train_ds, valid_ds = train_valid_split(dataset)

    train_loader = DataLoader(DroneAudioDataset(train_ds, cap_length=cap_length, keep_prob=keep_prob), batch_size=32)
    valid_loader = DataLoader(DroneAudioDataset(valid_ds, cap_length=cap_length, keep_prob=keep_prob), batch_size=32)

    first_batch = next(iter(train_loader))
    input_size = first_batch['x'].shape[1]  # number of features
    model = LogisticRegressionModel(input_dim = input_size)  

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_loader))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 1
    for epoch in range(num_epochs):
        train_acc, train_loss = single_epoch(model, train_loader, criterion, optimizer)
        valid_acc, valid_precision, valid_recall, valid_f1, valid_loss, (true_positives, false_positives, true_negatives, false_negatives) = evaluate(model, valid_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1 Score: {valid_f1:.4f}")
        print(f"Confusion Matrix: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")

    print(f"Train windows: {train_loader.dataset.drone_windows} drone, {train_loader.dataset.non_drone_windows} non-drone")
    print(f"Valid windows: {valid_loader.dataset.drone_windows} drone, {valid_loader.dataset.non_drone_windows} non-drone")

    train_window_diff = abs(train_loader.dataset.drone_windows - train_loader.dataset.non_drone_windows)
    valid_window_diff = abs(valid_loader.dataset.drone_windows - valid_loader.dataset.non_drone_windows)
    average_window_diff = (train_window_diff + valid_window_diff) / 2   

    return valid_f1, average_window_diff

if __name__ == "__main__":

    keep_prob = [1]
    cap_lengths = [13, 14, 15, 16, 17, 18, 19, 20]

    for cap_length in cap_lengths:
        for k in keep_prob:
            print(f"Testing cap_length={cap_length}, keep_prob={k}")
            valid_f1, avg_window_diff = main(cap_length=cap_length, keep_prob=k)

            print(f"F1 Score: {valid_f1:.4f}, Average Window Difference: {avg_window_diff:.2f} with cap_length={cap_length} and keep_prob={k}")

'''
Testing cap_length=10, keep_prob=0.85
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 454382.93it/s]
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 571950.55it/s]
Dataset loaded successfully!
Training label distribution: Counter({1: 130872, 0: 13383})
Validation label distribution: Counter({1: 32719, 0: 3346})
Epoch 1/1 - Train Loss: 2.2119, Train Accuracy: 0.9438
Epoch 1/1 - Valid Loss: 15.9583, Valid Accuracy: 0.7298, Precision: 0.6565, Recall: 0.9998, F1 Score: 0.7926
Confusion Matrix: TP=28176, FP=14740, TN=11645, FN=5
Train windows: 225537 drone, 210714 non-drone
Valid windows: 28181 drone, 26385 non-drone


Testing cap_length=10, keep_prob=0.8
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 554501.21it/s]
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 517651.44it/s]
Dataset loaded successfully!
Training label distribution: Counter({1: 130872, 0: 13383})
Validation label distribution: Counter({1: 32719, 0: 3346})
Epoch 1/1 - Train Loss: 2.2469, Train Accuracy: 0.9420
Epoch 1/1 - Valid Loss: 26.4586, Valid Accuracy: 0.6743, Precision: 0.6060, Recall: 0.9999, F1 Score: 0.7546
Confusion Matrix: TP=26555, FP=17266, TN=9193, FN=3
Train windows: 212146 drone, 210561 non-drone
Valid windows: 26558 drone, 26459 non-drone


Testing cap_length=15, keep_prob=1
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 498712.98it/s]
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 519294.78it/s]
Dataset loaded successfully!
Training label distribution: Counter({1: 130872, 0: 13383})
Validation label distribution: Counter({1: 32719, 0: 3346})
Epoch 1/1 - Train Loss: 2.2642, Train Accuracy: 0.9468
Epoch 1/1 - Valid Loss: 48.1356, Valid Accuracy: 0.6228, Precision: 0.5742, Recall: 0.9998, F1 Score: 0.7295
Confusion Matrix: TP=33329, FP=24716, TN=7494, FN=6
Train windows: 267764 drone, 258097 non-drone
Valid windows: 33335 drone, 32210 non-drone
F1 Score: 0.7295, Average Window Difference: 5396.00 with cap_length=15 and keep_prob=1


Testing cap_length=16, keep_prob=1
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 445716.23it/s]
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 539860.91it/s]
Dataset loaded successfully!
Training label distribution: Counter({1: 130872, 0: 13383})
Validation label distribution: Counter({1: 32719, 0: 3346})
Epoch 1/1 - Train Loss: 2.8758, Train Accuracy: 0.9466
Epoch 1/1 - Valid Loss: 59.9576, Valid Accuracy: 0.6091, Precision: 0.5613, Recall: 0.9999, F1 Score: 0.7190
Confusion Matrix: TP=33407, FP=26106, TN=7282, FN=2
Train windows: 268134 drone, 267496 non-drone
Valid windows: 33409 drone, 33388 non-drone
F1 Score: 0.7190, Average Window Difference: 329.50 with cap_length=16 and keep_prob=1

Testing cap_length=17, keep_prob=1
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 310984.52it/s]
Resolving data files: 100%|████████████████████████████████| 39/39 [00:00<00:00, 522612.96it/s]
Dataset loaded successfully!
Training label distribution: Counter({1: 130872, 0: 13383})
Validation label distribution: Counter({1: 32719, 0: 3346})
Epoch 1/1 - Train Loss: 2.4727, Train Accuracy: 0.9478
Epoch 1/1 - Valid Loss: 47.0141, Valid Accuracy: 0.7317, Precision: 0.6472, Recall: 0.9979, F1 Score: 0.7852
Confusion Matrix: TP=33369, FP=18190, TN=16428, FN=70
Train windows: 268592 drone, 276706 non-drone
Valid windows: 33439 drone, 34618 non-drone
F1 Score: 0.7852, Average Window Difference: 4646.50 with cap_length=17 and keep_prob=1

'''
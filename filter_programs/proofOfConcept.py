import os
import numpy as np
import librosa

# Paths (replace with wherever y'all are storing the files locally)
large_drone_path = r"C:\Users\justi\Documents\Coding Stuff\testStuff\DroneAudioDataset-master\Our_drones_mixed\Large"
small_drone_path = r"C:\Users\justi\Documents\Coding Stuff\testStuff\DroneAudioDataset-master\Our_drones_mixed\Small"
non_drone_path_one = r"C:\Users\justi\Documents\Coding Stuff\testStuff\DroneAudioDataset-master\Our_drones_mixed\unknown"
non_drone_path_two = r"C:\Users\justi\Documents\Coding Stuff\testStuff\DroneAudioDataset-master\Our_drones_mixed\non_drone"

# Feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        if len(y) < 2048:
            return None

        features = {}

        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        S = np.abs(librosa.stft(y))
        total_energy = np.sum(S)

        if total_energy < 1e-10:
            return None

        S_norm = S / total_energy
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
        features['spectral_entropy'] = entropy

        return features

    except:
        return None


# Classification rule
def is_drone(features):
    return (
        1650 <= features['spectral_bandwidth'] <= 2500 and
        1500 <= features['spectral_centroid'] <= 4600 and
        9 <= features['spectral_entropy'] <= 12 and
        4000 <= features['spectral_rolloff'] <= 7000
    )


# Counters
TP = FP = TN = FN = 0


def evaluate_folder(folder, true_label):
    global TP, FP, TN, FN

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)

            features = extract_features(path)

            if features is None:
                continue  # skip bad files

            prediction = is_drone(features)

            if prediction and true_label == "drone":
                TP += 1
            elif prediction and true_label == "non_drone":
                FP += 1
            elif not prediction and true_label == "non_drone":
                TN += 1
            elif not prediction and true_label == "drone":
                FN += 1


# Run evaluation
print("Evaluating large drones...")
evaluate_folder(large_drone_path, "drone")

print("Evaluating small drones...")
evaluate_folder(small_drone_path, "drone")

print("Evaluating non-drones...")
evaluate_folder(non_drone_path_one, "non_drone")
evaluate_folder(non_drone_path_two, "non_drone")

# Results
total = TP + FP + TN + FN

accuracy = (TP + TN) / total if total > 0 else 0

print("\n===== RESULTS =====")
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"\nTotal Samples: {total}")
print(f"Accuracy: {accuracy * 100:.2f}%")
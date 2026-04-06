import os
from datasets import load_dataset, Dataset, DatasetDict, Audio, Value, concatenate_datasets
import torchaudio
import torch
import numpy as np
from collections import Counter
from torch.utils.data import IterableDataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from amplify import amplify

SAMPLING_RATE = 16000  # 16 kHz
WINDOW_SIZE = 16000  # 1 second window
HOP_SIZE = 8000  # 0.5 second hop


class DroneAudioDataset(IterableDataset):
    def __init__(self, ds, window_size=WINDOW_SIZE, hop_size=HOP_SIZE, cap_length=None, train=True):
        self.ds = ds
        self.window_size = window_size
        self.hop_size = hop_size
        self.cap_length = cap_length
        self.non_drone_windows = 0
        self.small_drone_windows = 0
        self.large_drone_windows = 0
        self.train = train

        # initialize spectrograph and dB transforms
        self.spectrograph_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=64 )
        self.to_db_transform = torchaudio.transforms.AmplitudeToDB()

    def __iter__(self):
        """ 
        Yields spectrograph features and labels for each audio window in the dataset.
        """
        for sample in self.ds:
            waveform = sample['audio']['array']
            label = sample['label']
            windows = window_audio_samples(waveform, window=self.window_size, hop=self.hop_size, cap_length=self.cap_length)

            # windowing keep probabilities to balance classes and reduce overfitting on long audio samples
            if label == 0:
                keep_prob = 1
                windows = [w for w in windows if np.random.rand() < keep_prob]
                self.non_drone_windows += len(windows)

                if len(windows) == 0:
                    continue
            elif label == 1:
                keep_prob = 1
                windows = [w for w in windows if np.random.rand() < keep_prob]
                self.small_drone_windows += len(windows)

                if len(windows) == 0:
                    continue
            elif label == 2:
                keep_prob = 1
                windows = [w for w in windows if np.random.rand() < keep_prob]
                self.large_drone_windows += len(windows)

                if len(windows) == 0:
                    continue

            for window in windows:
                window = amplify(window, SAMPLING_RATE, train=self.train)
                window = rms_normalize_window(window, target_rms=0.1)

                x = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
                spectrograph = self.to_db_transform(self.spectrograph_transform(x)).squeeze(0)

                features = spectrograph.flatten().cpu().numpy()

                yield {'x': features, 'y': label}

def rms_normalize_window(window, target_rms=0.1):
    w = np.asarray(window, dtype=np.float32)
    rms = np.sqrt(np.mean(w**2) + 1e-12)
    rms = max(rms, 1e-6)
    scale = target_rms / rms
    w = w * scale
    w = np.clip(w, -1.0, 1.0)
    return w


def load_hf_non_drone_dataset():
    """
    Loads the drone audio detection dataset from Hugging Face Datasets library.
    """
    huggingface_ds = load_dataset("geronimobasso/drone-audio-detection-samples")
    print("Hugging Face Dataset loaded successfully!")

    train_ds = huggingface_ds['train']
    none_drone_ds = train_ds.filter(lambda x: x['label'] == 0)
    none_drone_ds = none_drone_ds.cast_column('audio', Audio())

    none_drone_ds = none_drone_ds.cast_column('label', Value('int64'))

    return none_drone_ds

def load_local_drone_audio_dataset(local_dir="/home/luke_gut/Drone-Detection/drone_data"):
    """ Loads the local drone audio dataset from the specified directory.
    The directory should have subfolders for each class (large, medium, small, non_drone) containing the respective audio files.
    Args:
        local_dir (str): The path to the local dataset directory.
    """
    label_map = {
        'large': 2,
        'small': 1,
        'non_drone': 0
    }

    audio_paths = []
    labels = []
    for class_folder, label in label_map.items():
        class_dir = os.path.join(local_dir, class_folder)

        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(root, file))
                    labels.append(label)
    
    local_ds = Dataset.from_dict({'audio': audio_paths, 'label': labels})
    local_ds = local_ds.cast_column('audio', Audio())
    print("Local drone audio dataset loaded successfully!")
    return local_ds

def aggregate_datasets(ds1, ds2):
    """ Combines two datasets into a single DatasetDict with a 'train' split.
    
    Args:
        ds1 (Dataset): The first dataset to combine.
        ds2 (Dataset): The second dataset to combine.
    """
    combined = concatenate_datasets([ds1, ds2])

    combined_ds = DatasetDict({
        "train": combined
    })
    return combined_ds


def train_valid_split(ds, valid_ratio=0.2):
    """ Splits the dataset into training and validation sets while maintaining class balance.
    
    Args:
        ds (DatasetDict): The input dataset
        valid_ratio (float): The proportion of the dataset to include in the validation split.
    """
    labels = ds['train']['label']

    # Create a list of indices for each class
    class_indices = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Split indices for each class into train and validation sets
    train_indices = []
    valid_indices = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - valid_ratio))
        train_indices.extend(indices[:split_point])
        valid_indices.extend(indices[split_point:])

    # Create train and validation datasets
    train_ds = ds['train'].select(train_indices)
    valid_ds = ds['train'].select(valid_indices)

    # Print label distribution in train and validation sets
    print(f"Training label distribution: {Counter(train_ds['label'])}")
    print(f"Validation label distribution: {Counter(valid_ds['label'])}")

    return train_ds, valid_ds


def window_audio_samples(waveform, window = WINDOW_SIZE, hop=HOP_SIZE, cap_length=None):
    """ Splits the audio waveform into overlapping windows.
    
    Args:
        waveform (array): The input audio waveform as an array of floats.
        window (int): The size of each window in samples.
        hop (int): The hop size between windows in samples.
        cap_length (int, optional): Maximum number of windows to return for long audio. If None, returns all windows.
    """
        
    waveform = np.asarray(waveform, dtype=np.float32)

    # short waveform, pad with zeros
    if len(waveform) <= window:
        padding = window - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
        return [waveform]

    # long waveform, split into windows
    windows = [waveform[i:i+window] for i in range(0, len(waveform) - window + 1, hop)]

    # cap windows from long audio to a cap_length if specified
    if cap_length is not None and len(windows) > cap_length:
        idx = np.random.choice(len(windows), cap_length, replace=False)
        windows = [windows[i] for i in idx]

    return windows


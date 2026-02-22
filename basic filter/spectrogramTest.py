from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torchaudio

SAMPLING_RATE = 16000   # 16k Hz
WINDOW_SIZE = 9600      # 1s window size
HOP_SIZE = 8000         # only hop 0.5s so windows overlap

# lower and upper bound of frequencies where drones are most likely to appear
DRONE_LOW = 175
DRONE_HIGH = 425

# create spectrogram of same type as training spectrograms
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLING_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=64
)
db_transform = torchaudio.transforms.AmplitudeToDB()

# split the audio files into 0.6s segments to turn long files into multiple samples
def window_audio_samples(waveform, window=WINDOW_SIZE, hop=HOP_SIZE):
    waveform = np.asarray(waveform, dtype=np.float32)

    if len(waveform) <= window:
        waveform = np.pad(waveform, (0, window - len(waveform)))
        return [waveform]

    return [
        waveform[i:i+window]
        for i in range(0, len(waveform) - window + 1, hop)
    ]

# run a fourier transform to see if the frequency with the highest energy is between the preset bounds
def detect_drone_window(signal, fs):
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)

    peak_index = np.argmax(power)
    peak_freq = freqs[peak_index]

    return DRONE_LOW <= peak_freq <= DRONE_HIGH

# hopefully identical style spectrogram to send to ML algorithm
def save_spectrogram(signal, fs, filename="detected_drone_spectrogram.png"):
    x = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0)

    spectrograph = db_transform(mel_transform(x)).squeeze(0)
    spectrogram_np = spectrograph.numpy()

    plt.figure()
    plt.imshow(spectrogram_np, origin='lower', aspect='auto')
    plt.colorbar(label="dB")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bins")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Mel spectrogram saved as {filename}")


if __name__ == "__main__":
    ds = load_dataset("geronimobasso/drone-audio-detection-samples")
    print("Dataset loaded!")

    falsePositives = 0
    totalWindows = 0

    for idx, sample in enumerate(ds['train']):
        label = sample['label']
        waveform = sample['audio']['array']

        # create spectrogram if sample contains drone
        if label == 1:
            detected = detect_drone_window(waveform, SAMPLING_RATE)

            if detected:
                print(f"Drone detected in full drone sample at index {idx}")
                print(f"False positives: {falsePositives}")
                save_spectrogram(waveform, SAMPLING_RATE)
                sys.exit(0)

        # split non drone samples and count false positives
        else:
            windows = window_audio_samples(waveform)

            for w in windows:
                totalWindows += 1
                detected = detect_drone_window(w, SAMPLING_RATE)

                if detected:
                    print(f"\rDrone detected in non-drone sample at index {idx}", end="", flush=True)
                    falsePositives += 1

    print(f"False positives: {falsePositives}")
    print(f"True negatives: {totalWindows-falsePositives}")
    print("No drone samples detected.")
from datasets import load_dataset
import numpy as np

SAMPLING_RATE = 16000   # 16k Hz
WINDOW_SIZE = 9600      # 0.6s window size
HOP_SIZE = 8000         # only hop 0.5s so windows slightly overlap

# lower and upper bound of frequencies where drones are most likely to appear
DRONE_LOW = 175
DRONE_HIGH = 425

# split the audio files into 0.6s segments to turn long files into multiple samples
def windowAudioSamples(waveform, window=WINDOW_SIZE, hop=HOP_SIZE):
    waveform = np.asarray(waveform, dtype=np.float32)

    if len(waveform) <= window:
        waveform = np.pad(waveform, (0, window - len(waveform)))
        return [waveform]

    return [
        waveform[i:i+window]
        for i in range(0, len(waveform) - window + 1, hop)
    ]

# run a fourier transform to see if the frequency with the highest energy is between the preset bounds
def detectDroneWindow(signal, fs):
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)

    peakIndex = np.argmax(power)
    peakFreq = freqs[peakIndex]

    return DRONE_LOW <= peakFreq <= DRONE_HIGH

if __name__ == "__main__":
    ds = load_dataset("geronimobasso/drone-audio-detection-samples")
    print("Dataset loaded!")

    totalDroneSamples = 0
    totalNonDroneSamples = 0
    falsePositives = 0
    falseNegatives = 0

    for idx, sample in enumerate(ds['train']):
        label = sample['label']
        waveform = sample['audio']['array']

        # check if sample with drone is detected, count false negatives
        if label == 1:
            totalDroneSamples += 1

            detected = detectDroneWindow(waveform, SAMPLING_RATE)

            if not detected:
                falseNegatives += 1

        # check if sample without drone is ignored, count false positives
        else:
            windows = windowAudioSamples(waveform)

            for w in windows:
                totalNonDroneSamples += 1

                detected = detectDroneWindow(w, SAMPLING_RATE)

                if detected:
                    falsePositives += 1

        if (idx + 1) % 100 == 0:
            print(f"\rProcessed {idx + 1} samples", end="", flush=True)

    print(f"Total drone files: {totalDroneSamples}")
    print(f"Total non-drone windows: {totalNonDroneSamples}")
    print(f"False positives: {falsePositives}")
    print(f"False negatives: {falseNegatives}")


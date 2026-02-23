from datasets import load_dataset
from logistic_regression import LogisticRegressionModel
import torch
import torch.nn as nn
from data_loader import window_audio_samples, load_drone_audio_dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from amplify import amplify

SAMPLING_RATE = 16000  # 16 kHz
WINDOW_SIZE = 16000  # 1 second window
HOP_SIZE = 8000  # 0.5 second hop


def extract_features_and_window(waveform):

    windows = window_audio_samples(waveform, window=WINDOW_SIZE, hop=HOP_SIZE)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE, n_fft=1024, hop_length=256, n_mels=64)
    to_db =  torchaudio.transforms.AmplitudeToDB()

    features_list = []
    for window in windows:
        window = amplify(window, SAMPLING_RATE, train=False)

        x = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        spectrograph = to_db(mel(x)).squeeze(0)
        features = spectrograph.flatten().cpu().numpy()
        features_list.append(features)

    return torch.as_tensor(features_list, dtype=torch.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("best_logistic_model.pth", map_location=device)
    input_dim = checkpoint["input_dim"]

    model = LogisticRegressionModel(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    
    waveform = torch.randn(SAMPLING_RATE * 5)  # example 5-second audio clip

    features = extract_features_and_window(waveform).to(device)

    with torch.no_grad():
        logits = model(features.float())
        probabilities = torch.sigmoid(logits).view(-1)
    
    total_prob = probabilities.mean().item()
    prediction = 1 if total_prob >= 0.5 else 0

    print(f"Predicted class: {prediction} (probability: {total_prob:.4f})")


if __name__ == "__main__":
    main()



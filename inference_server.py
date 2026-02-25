from datasets import load_dataset
from logistic_regression import LogisticRegressionModel
import torch
import torch.nn as nn
from data_loader import window_audio_samples, load_drone_audio_dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from amplify import amplify

import threading
import time
import tkinter as tk


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

class StatusWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Drone Detector")
        self.root.geometry("800x400")
        self.root.resizable(False, False)

        self.label = tk.Label(
            self.root,
            text="Starting…",
            font=("Helvetica", 25, "bold"),
            fg="black"
        )
        self.label.pack(expand=True, fill="both")

        self._flashing = False
        self._flash_state = False

    def start_flashing_yellow(self, text="Potential drone detected…"):
        self._flashing = True
        self.label.config(text=text)
        self._flash_yellow()

    def _flash_yellow(self):
        if not self._flashing:
            return
        self._flash_state = not self._flash_state
        bg = "yellow" if self._flash_state else "black"
        fg = "black" if self._flash_state else "yellow"
        self.root.configure(bg=bg)
        self.label.configure(bg=bg, fg=fg)
        self.root.after(200, self._flash_yellow)

    def stop_flashing(self):
        self._flashing = False

    def set_result(self, prediction: int):
        self.stop_flashing()
        if prediction == 1:
            bg = "green"
            msg = "No drone detected"
        else:
            bg = "red"
            msg = "Drone detected"

        self.root.configure(bg=bg)
        self.label.configure(bg=bg, text=msg, fg="white")

    def run(self):
        self.root.mainloop()

def run_inference_loop(status_window: StatusWindow):
    status_window.start_flashing_yellow()
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
    status_window.root.after(0, lambda: status_window.set_result(prediction))

def main():
    status_window = StatusWindow()
    inference_thread = threading.Thread(target=run_inference_loop, args=(status_window,), daemon=True)
    inference_thread.start()
    status_window.run()


if __name__ == "__main__":
    main()



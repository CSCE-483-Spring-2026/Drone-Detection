from datasets import load_dataset
from logistic_regression import LogisticRegressionModel
import torch
import torch.nn as nn
from data_loader import window_audio_samples, load_drone_audio_dataset, rms_normalize_window
from audio_capture import record_audio
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from amplify import amplify
import numpy as np

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
        window = rms_normalize_window(window, target_rms=0.1)

        x = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)
        spectrograph = to_db(mel(x)).squeeze(0)

        features = spectrograph.flatten().cpu().numpy()
        features_list.append(features)

    features_np = np.stack(features_list, axis=0)
    return torch.from_numpy(features_np).float()

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

    def start_flashing_yellow(self, text="Potential drone detected, analyzing…"):
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
        if prediction == 0:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("best_logistic_model.pth", map_location=device)
    input_dim = checkpoint["input_dim"]

    model = LogisticRegressionModel(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    feat_mean = checkpoint.get("feat_mean", None)
    feat_std = checkpoint.get("feat_std", None)

    if feat_mean is not None and feat_std is not None:
        feat_mean = feat_mean.to(device)
        feat_std = feat_std.to(device)

    model.to(device)
    model.eval()


    w = model.linear.weight.detach().cpu()
    b = model.linear.bias.detach().cpu()
    print("w min/max:", w.min().item(), w.max().item())
    print("w L2 norm:", w.norm().item())
    print("bias:", b.item())
    

    status_window.start_flashing_yellow()
    wave_path = record_audio("capture.wav", duration=4, sample_rate=SAMPLING_RATE)
    waveform, sr = torchaudio.load(wave_path, normalize=True)

    if sr != SAMPLING_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLING_RATE)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]

    # test waveform from drone audio dataset
    # dataset = load_drone_audio_dataset()
    # sample = dataset['train'][-150]
    # waveform = torch.as_tensor(sample['audio']['array'], dtype=torch.float32)

    features = extract_features_and_window(waveform.cpu().numpy()).to(device)
    if feat_mean is not None and feat_std is not None:
        features = (features - feat_mean) / (feat_std + 1e-8)

    print("num windows:", features.shape[0])
    print("features per window:", features.shape[1], "expected:", input_dim)

    with torch.no_grad():
        logits = model(features.float()) 
        logits = logits.view(-1)   
        probabilities = torch.sigmoid(logits)  
        
        
    total_logit = logits.mean() 
    total_prob = torch.sigmoid(total_logit).item()
    prediction = int(total_prob >= 0.75)

    print(f"Predicted class: {prediction} (probability: {total_prob:.4f})")
    print("logits/probabilities for each window:", list(zip(logits.cpu().numpy(), probabilities.cpu().numpy())))
    print("feature stats:", features.min().item(), features.max().item(), features.mean().item(), features.std().item())
    print("logit stats:", logits.min().item(), logits.max().item())

    status_window.root.after(0, lambda: status_window.set_result(prediction))

def main():
    status_window = StatusWindow()
    inference_thread = threading.Thread(target=run_inference_loop, args=(status_window,), daemon=True)
    inference_thread.start()
    status_window.run()


if __name__ == "__main__":
    main()



# from datasets import load_dataset
from .logistic_regression import LogisticRegressionModel
import torch
import torch.nn as nn
from .data_loader import window_audio_samples, rms_normalize_window#, load_drone_audio_dataset
# from audio_capture import record_audio # for testing with live audio capture, not used in current version
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from .amplify import amplify
import numpy as np
import tkinter as tk
import soundfile as sf


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

        # Track scheduled reset so we can cancel it if needed
        self._reset_after_id = None

        # Set initial resting state
        self.set_resting_state()

    def set_resting_state(self, text="Nothing detected"):
        self.stop_flashing()
        if self._reset_after_id is not None:
            self.root.after_cancel(self._reset_after_id)
            self._reset_after_id = None

        bg = "black"
        fg = "white"
        self.root.configure(bg=bg)
        self.label.configure(bg=bg, fg=fg, text=text)

    def start_flashing_yellow(self, text="Potential drone detected, analyzing…"):
        # If we were going to reset, cancel that because we're actively analyzing now
        if self._reset_after_id is not None:
            self.root.after_cancel(self._reset_after_id)
            self._reset_after_id = None

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

    def set_result(self, prediction: int, hold_ms: int = 3000):
        """
        Show Drone/No drone result, hold for hold_ms, then return to resting state.
        """
        self.stop_flashing()

        if prediction == 0:
            bg = "green"
            msg = "No drone detected"
        else:
            bg = "red"
            msg = "Drone detected"

        self.root.configure(bg=bg)
        self.label.configure(bg=bg, text=msg, fg="white")

        # Schedule return to resting state
        if self._reset_after_id is not None:
            self.root.after_cancel(self._reset_after_id)

        self._reset_after_id = self.root.after(
            hold_ms,
            lambda: self.set_resting_state("Nothing detected. Waiting for audio..")
        )

    def run(self):
        self.root.mainloop()

class InferenceEngine:
    def __init__(self, model_path="./prediction/best_logistic_model.pth", threshold=0.63, device=None, detailed_logging=False):  
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.detailed_logging = detailed_logging
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]

        self.model = LogisticRegressionModel(input_dim=checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.feat_mean = checkpoint.get("feat_mean", None)
        self.feat_std = checkpoint.get("feat_std", None)
        if self.feat_mean is not None and self.feat_std is not None:
            self.feat_mean = self.feat_mean.to(self.device)
            self.feat_std = self.feat_std.to(self.device)

        if self.detailed_logging:
            w = self.model.linear.weight.detach().cpu()
            b = self.model.linear.bias.detach().cpu()
            print("w min/max:", w.min().item(), w.max().item())
            print("w L2 norm:", w.norm().item())
            print("bias:", b.item())
    
    def load_waveform(self, wave_path):
        audio, sr = sf.read(wave_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)

        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLING_RATE)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]

        return waveform

    @torch.no_grad()
    def predict(self, wave_path):
        waveform = self.load_waveform(wave_path)

        features = extract_features_and_window(waveform.cpu().numpy()).to(self.device)

        if self.feat_mean is not None and self.feat_std is not None:
            features = (features - self.feat_mean) / (self.feat_std + 1e-8)

        logits = self.model(features.float()).view(-1)
        probabilities = torch.sigmoid(logits)

        total_logit = logits.mean()
        total_prob = torch.sigmoid(total_logit).item()
        prediction = int(total_prob >= self.threshold)

        if self.detailed_logging:
            print("num windows:", features.shape[0])
            print("features per window:", features.shape[1], "expected:", self.model.linear.in_features)
            print("feature stats:", features.min().item(), features.max().item(), features.mean().item(), features.std().item())
            print("logit stats:", logits.min().item(), logits.max().item())
        
        return {"prediction": prediction, "probability": total_prob, "logits": logits, "probabilities": probabilities, "num_windows": int(features.shape[0])}


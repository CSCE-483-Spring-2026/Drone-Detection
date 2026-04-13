import threading

from cnn_training import CNNModel
import torch
import torch.nn as nn
from data_loader import window_audio_samples, rms_normalize_window
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from amplify import amplify
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
        elif prediction == 1:
            bg = "red"
            msg = "Small drone detected"
        else:
            bg = "red"
            msg = "Large drone detected"

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
    def __init__(self, model_path="./best_cnn_model.pth", device=None, detailed_logging=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detailed_logging = detailed_logging

        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_h = checkpoint["spec_h"]
        self.input_w = checkpoint["spec_w"]
        self.num_classes = checkpoint["num_classes"]

        self.model = CNNModel(input_channels=1, num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.feat_mean = checkpoint.get("feat_mean", None)
        self.feat_std = checkpoint.get("feat_std", None)

        if self.feat_mean is not None and self.feat_std is not None:
            self.feat_mean = self.feat_mean.to(self.device)
            self.feat_std = self.feat_std.to(self.device)

        if self.detailed_logging:
            print(
                f"Model loaded on {self.device} with input shape "
                f"({self.input_h}, {self.input_w}) and {self.num_classes} classes.")

    def load_waveform(self, wave_path):
        audio, sr = sf.read(wave_path, always_2d=False)

        audio = torch.as_tensor(audio, dtype=torch.float32)

        if audio.ndim == 2:
            audio = audio.mean(dim=1)

        waveform = audio

        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLING_RATE)

        if self.detailed_logging:
            print(f"Loaded waveform shape: {waveform.shape}, original sr={sr}")

        return waveform

    @torch.inference_mode()
    def predict(self, features):
        features = features.to(self.device)

        expected_dim = self.input_h * self.input_w
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(f"Feature shape mismatch. Got {tuple(features.shape)}, expected (num_windows, {expected_dim}).")

        if self.feat_mean is not None and self.feat_std is not None:
            features = (features - self.feat_mean) / (self.feat_std + 1e-8)

        if self.detailed_logging:
            print(f"Normalized feature shape: {features.shape}")

        features = features.view(-1, 1, self.input_h, self.input_w)

        if self.detailed_logging:
            print(f"CNN input shape: {features.shape}")

        outputs = self.model(features)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        if self.detailed_logging:
            print(f"Output shape: {outputs.shape}")
            print(f"Predicted classes: {predicted_classes.tolist()}")

        return predicted_classes.cpu().numpy(), probabilities.cpu().numpy()

def saved_file_predict(status_window, wave_path):
    """
    Load a saved audio file, run inference on it, and show the result
    in the status window. This does NOT record new audio.
    """
    try:
        engine = InferenceEngine(detailed_logging=True)

        # Update UI to show analysis has started
        status_window.root.after(
            0,
            lambda: status_window.start_flashing_yellow(
                f"Analyzing saved file:\n{wave_path}"
            )
        )

        # Load waveform from disk
        waveform = engine.load_waveform(wave_path)

        if waveform.numel() == 0:
            raise ValueError("Loaded waveform is empty.")

        # Extract per-window features
        features = extract_features_and_window(waveform)

        if features.numel() == 0:
            raise ValueError("No windows/features were extracted from the audio file.")

        # Predict each window
        predicted_classes, probabilities = engine.predict(features)

        # Aggregate window-level probabilities into one clip-level prediction
        mean_probs = probabilities.mean(axis=0)
        final_prediction = int(np.argmax(mean_probs))

        print(f"Window predictions: {predicted_classes.tolist()}")
        print(f"Mean probabilities: {mean_probs}")
        print(f"Final prediction: {final_prediction}")

        # Update UI safely from Tkinter main thread
        status_window.root.after(0, lambda: status_window.set_result(final_prediction))

    except Exception as e:
        print(f"[ERROR] saved_file_predict failed: {e}")
        status_window.root.after(
            0,
            lambda: status_window.set_resting_state(f"Error:\n{str(e)}")
        )

def main():
    status_window = StatusWindow()

    inference_thread = threading.Thread(target=saved_file_predict, args=(status_window, '/home/luke_gut/Drone-Detection/cnn_prediction/unknown/clip_nondrone_001_20260410_110105.wav'), daemon=True)
    inference_thread.start()
    
    status_window.run()

if __name__ == "__main__":
    main()
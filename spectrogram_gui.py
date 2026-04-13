import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cnn_prediction.amplify import amplify
from cnn_prediction.data_loader import load_drone_audio_dataset, window_audio_samples, SAMPLING_RATE

BG = "#050814"
PANEL = "#0a1024"
PANEL2 = "#070b18"
GRID = "#1a2a4a"
CYAN = "#00eaff"
MAG = "#b100ff"
TXT = "#d9e2ff"
MUTED = "#7f94c7"
BTN = "#0b1533"

WINDOW_SIZE = 16000
HOP_SIZE = 8000
CAP_WINDOWS = 10

def _to_1d_torch(x):
    if isinstance(x, np.ndarray):
        t = torch.as_tensor(x, dtype=torch.float32)
    elif torch.is_tensor(x):
        t = x.to(dtype=torch.float32)
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 1:
        t = t.reshape(-1)
    return t

def rms_db(x):
    x = _to_1d_torch(x)
    r = torch.sqrt(torch.mean(x * x) + 1e-12)
    return float(20.0 * torch.log10(r + 1e-12))

def peak_abs(x):
    x = _to_1d_torch(x)
    return float(torch.max(torch.abs(x)).item())

def fft_db(x, sr):
    x = _to_1d_torch(x)
    x = x - x.mean()
    n = x.numel()
    if n < 16:
        freqs = torch.fft.rfftfreq(max(n, 16), d=1.0 / sr)
        db = torch.zeros_like(freqs)
        return freqs.cpu().numpy(), db.cpu().numpy()
    w = torch.hann_window(n, device=x.device, dtype=x.dtype)
    X = torch.fft.rfft(x * w)
    mag = torch.abs(X) + 1e-12
    db = 20.0 * torch.log10(mag)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr)
    return freqs.cpu().numpy(), db.cpu().numpy()

def band_db_power(x, sr, f_lo, f_hi):
    x = _to_1d_torch(x)
    x = x - x.mean()
    n = x.numel()
    if n < 16:
        return -120.0
    w = torch.hann_window(n, device=x.device, dtype=x.dtype)
    X = torch.fft.rfft(x * w)
    p = (X.real * X.real + X.imag * X.imag) + 1e-12
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    band_power = torch.sum(p[mask]) + 1e-12
    return float(10.0 * torch.log10(band_power).item())

class SpecGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Audio Analyzer")
        self.root.configure(bg=BG)

        self.ds = load_drone_audio_dataset()["train"]
        self.sample_idx = 0
        self.window_idx = 0
        self.windows = []
        self.label = 0
        self.before = None
        self.after = None
        self.dur = 1.0

        try:
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE, n_fft=1024, hop_length=256, n_mels=128)
        except TypeError:
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE, hop_length=256, n_mels=128)
        self.db = torchaudio.transforms.AmplitudeToDB()

        self._build_ui()
        self._load_sample(0)

    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.side = tk.Frame(self.root, bg=PANEL2, width=280)
        self.side.grid(row=0, column=0, sticky="nsw")
        self.side.grid_propagate(False)

        title = tk.Label(self.side, text="DRONE AUDIO\nANALYZER", fg=CYAN, bg=PANEL2, font=("Consolas", 18, "bold"), justify="left")
        title.pack(anchor="nw", padx=14, pady=(16, 8))

        self.meta = tk.Label(self.side, text="", fg=MUTED, bg=PANEL2, font=("Consolas", 10), justify="left")
        self.meta.pack(anchor="nw", padx=14, pady=(0, 10))

        tk.Frame(self.side, bg=GRID, height=1).pack(fill="x", padx=14, pady=6)

        tk.Label(self.side, text="NAVIGATION", fg=TXT, bg=PANEL2, font=("Consolas", 11, "bold")).pack(anchor="nw", padx=14, pady=(8, 6))

        row = tk.Frame(self.side, bg=PANEL2)
        row.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(row, text="Sample #", fg=TXT, bg=PANEL2, font=("Consolas", 10)).pack(side="left")
        self.sample_entry = tk.Entry(row, bg=BG, fg=TXT, insertbackground=TXT, relief="flat", width=10, font=("Consolas", 10))
        self.sample_entry.pack(side="left", padx=(8, 6))
        tk.Button(row, text="Go", command=self._go_sample, bg=BTN, fg=TXT, relief="flat", padx=10, font=("Consolas", 10, "bold")).pack(side="left")

        row2 = tk.Frame(self.side, bg=PANEL2)
        row2.pack(fill="x", padx=14, pady=(0, 10))
        tk.Button(row2, text="Prev Sample", command=self._prev_sample, bg=BTN, fg=TXT, relief="flat", font=("Consolas", 10, "bold")).pack(side="left", expand=True, fill="x", padx=(0, 6))
        tk.Button(row2, text="Next Sample", command=self._next_sample, bg=BTN, fg=TXT, relief="flat", font=("Consolas", 10, "bold")).pack(side="left", expand=True, fill="x")

        tk.Label(self.side, text="Window", fg=TXT, bg=PANEL2, font=("Consolas", 10)).pack(anchor="nw", padx=14)
        self.win_slider = ttk.Scale(self.side, from_=0, to=1, orient="horizontal", command=self._slider_window)
        self.win_slider.pack(fill="x", padx=14, pady=(4, 8))

        row3 = tk.Frame(self.side, bg=PANEL2)
        row3.pack(fill="x", padx=14, pady=(0, 10))
        tk.Button(row3, text="Prev Window", command=self._prev_window, bg=BTN, fg=TXT, relief="flat", font=("Consolas", 10, "bold")).pack(side="left", expand=True, fill="x", padx=(0, 6))
        tk.Button(row3, text="Next Window", command=self._next_window, bg=BTN, fg=TXT, relief="flat", font=("Consolas", 10, "bold")).pack(side="left", expand=True, fill="x")

        tk.Frame(self.side, bg=GRID, height=1).pack(fill="x", padx=14, pady=10)

        tk.Label(self.side, text="HOVER INSPECTOR", fg=TXT, bg=PANEL2, font=("Consolas", 11, "bold")).pack(anchor="nw", padx=14, pady=(0, 6))
        self.hover = tk.Label(self.side, text="Hover over a spectrogram", fg=CYAN, bg=PANEL2, font=("Consolas", 9), justify="left", wraplength=245)
        self.hover.pack(anchor="nw", padx=14)

        tk.Frame(self.side, bg=GRID, height=1).pack(fill="x", padx=14, pady=10)

        tk.Label(self.side, text="CREDIBILITY CHECK", fg=TXT, bg=PANEL2, font=("Consolas", 11, "bold")).pack(anchor="nw", padx=14, pady=(0, 6))
        self.cred = tk.Label(self.side, text="", fg=TXT, bg=PANEL2, font=("Consolas", 9), justify="left", wraplength=245)
        self.cred.pack(anchor="nw", padx=14)

        self.main = tk.Frame(self.root, bg=BG)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(12, 7), dpi=100, facecolor=BG)
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1.0, 0.70], width_ratios=[1.0, 0.05, 1.0, 0.05], wspace=0.28, hspace=0.34)

        self.ax_sb = self.fig.add_subplot(gs[0, 0])
        self.ax_sa = self.fig.add_subplot(gs[0, 2])
        self.cax_sb = self.fig.add_subplot(gs[0, 1])
        self.cax_sa = self.fig.add_subplot(gs[0, 3])
        self.ax_fft = self.fig.add_subplot(gs[1, 0])
        self.ax_wav = self.fig.add_subplot(gs[1, 2])

        for ax in [self.ax_sb, self.ax_sa, self.ax_fft, self.ax_wav, self.cax_sb, self.cax_sa]:
            ax.set_facecolor(PANEL)

        for ax in [self.ax_sb, self.ax_sa, self.ax_fft, self.ax_wav]:
            ax.grid(True, color=GRID, alpha=0.35, linewidth=0.8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)

        self.ax_sb.set_title("SPECTROGRAM — BEFORE", color=CYAN, fontweight="bold")
        self.ax_sa.set_title("SPECTROGRAM — AFTER", color=MAG, fontweight="bold")
        self.ax_fft.set_title("FFT MAGNITUDE — BEFORE vs AFTER", color=CYAN, fontweight="bold")
        self.ax_wav.set_title("WAVEFORM — BEFORE vs AFTER", color=MAG, fontweight="bold")

        self.ax_sb.set_xlabel("Time (s)", color=MUTED)
        self.ax_sa.set_xlabel("Time (s)", color=MUTED)
        self.ax_sb.set_ylabel("Mel bin (low→high)", color=MUTED)
        self.ax_sa.set_ylabel("Mel bin (low→high)", color=MUTED)

        self.ax_fft.set_xlabel("Frequency (Hz)", color=MUTED)
        self.ax_fft.set_ylabel("Magnitude (dB)", color=MUTED)
        self.ax_wav.set_xlabel("Samples", color=MUTED)
        self.ax_wav.set_ylabel("Amplitude", color=MUTED)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.im_b = None
        self.im_a = None
        self.cb_b = None
        self.cb_a = None

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _to_spec_db(self, wav_1d):
        x = _to_1d_torch(wav_1d).unsqueeze(0)
        s = self.db(self.mel(x)).squeeze(0)
        return s

    def _as_list(self, windows):
        if windows is None:
            return []
        if isinstance(windows, list):
            return windows
        if isinstance(windows, tuple):
            return list(windows)
        if isinstance(windows, np.ndarray):
            if windows.ndim == 1:
                return [windows]
            return [windows[i] for i in range(windows.shape[0])]
        if torch.is_tensor(windows):
            if windows.ndim == 1:
                return [windows.detach().cpu()]
            return [windows[i].detach().cpu() for i in range(windows.shape[0])]
        try:
            return list(windows)
        except Exception:
            return []

    def _load_sample(self, idx):
        idx = int(max(0, min(idx, len(self.ds) - 1)))
        self.sample_idx = idx
        sample = self.ds[idx]
        self.label = int(sample["label"])
        wav = sample["audio"]["array"]
        self.windows = self._as_list(window_audio_samples(wav, window=WINDOW_SIZE, hop=HOP_SIZE, cap_length=CAP_WINDOWS))
        if len(self.windows) == 0:
            self.windows = [np.zeros((WINDOW_SIZE,), dtype=np.float32)]
        self.window_idx = int(min(self.window_idx, len(self.windows) - 1))
        self.win_slider.configure(to=max(len(self.windows) - 1, 1))
        self.win_slider.set(self.window_idx)
        self._render()

    def _get_current_window(self):
        if len(self.windows) == 0:
            return np.zeros((WINDOW_SIZE,), dtype=np.float32)
        self.window_idx = int(max(0, min(self.window_idx, len(self.windows) - 1)))
        return self.windows[self.window_idx]

    def _render(self):
        self.before = self._get_current_window()
        self.after = amplify(self.before, SAMPLING_RATE, train=True)

        spec_b = self._to_spec_db(self.before).cpu().numpy()
        spec_a = self._to_spec_db(self.after).cpu().numpy()

        self.dur = float(len(_to_1d_torch(self.before)) / SAMPLING_RATE)

        vmin = -80.0
        vmax = 0.0

        if self.im_b is None:
            self.im_b = self.ax_sb.imshow(spec_b, origin="lower", aspect="auto", extent=[0.0, self.dur, 0, spec_b.shape[0]], vmin=vmin, vmax=vmax, cmap="magma")
            self.im_a = self.ax_sa.imshow(spec_a, origin="lower", aspect="auto", extent=[0.0, self.dur, 0, spec_a.shape[0]], vmin=vmin, vmax=vmax, cmap="magma")
            self.cb_b = self.fig.colorbar(self.im_b, cax=self.cax_sb)
            self.cb_a = self.fig.colorbar(self.im_a, cax=self.cax_sa)
            self.cb_b.ax.tick_params(colors=MUTED, labelsize=9)
            self.cb_a.ax.tick_params(colors=MUTED, labelsize=9)
            self.cb_b.set_label("dB", color=MUTED)
            self.cb_a.set_label("dB", color=MUTED)
        else:
            self.im_b.set_data(spec_b)
            self.im_a.set_data(spec_a)
            self.im_b.set_extent([0.0, self.dur, 0, spec_b.shape[0]])
            self.im_a.set_extent([0.0, self.dur, 0, spec_a.shape[0]])
            self.im_b.set_clim(vmin=vmin, vmax=vmax)
            self.im_a.set_clim(vmin=vmin, vmax=vmax)
            self.cb_b.update_normal(self.im_b)
            self.cb_a.update_normal(self.im_a)

        self.ax_sb.set_xlim(0.0, self.dur)
        self.ax_sa.set_xlim(0.0, self.dur)
        self.ax_sb.set_ylim(0, spec_b.shape[0])
        self.ax_sa.set_ylim(0, spec_a.shape[0])

        b = _to_1d_torch(self.before).cpu().numpy()
        a = _to_1d_torch(self.after).cpu().numpy()

        self.ax_fft.cla()
        self.ax_wav.cla()

        for ax in [self.ax_fft, self.ax_wav]:
            ax.set_facecolor(PANEL)
            ax.grid(True, color=GRID, alpha=0.35, linewidth=0.8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)

        freq_b, db_b = fft_db(b, SAMPLING_RATE)
        freq_a, db_a = fft_db(a, SAMPLING_RATE)

        hp = 50.0
        lp = 8000.0
        nyq = float(SAMPLING_RATE) / 2.0 - 1.0
        lp_eff = float(min(lp, nyq))

        self.ax_fft.plot(freq_b, db_b, color=CYAN, linewidth=1.0, label="before")
        self.ax_fft.plot(freq_a, db_a, color=MAG, linewidth=1.0, label="after")
        self.ax_fft.axvline(hp, color=TXT, linewidth=1.0, alpha=0.65)
        self.ax_fft.axvline(lp_eff, color=TXT, linewidth=1.0, alpha=0.65)
        self.ax_fft.set_xlim(0, min(12000.0, nyq))
        self.ax_fft.set_title("FFT MAGNITUDE — BEFORE vs AFTER", color=CYAN, fontweight="bold")
        self.ax_fft.set_xlabel("Frequency (Hz)", color=MUTED)
        self.ax_fft.set_ylabel("Magnitude (dB)", color=MUTED)
        leg = self.ax_fft.legend(frameon=True)
        leg.get_frame().set_facecolor(PANEL)
        leg.get_frame().set_edgecolor(GRID)

        self.ax_wav.plot(b, color=CYAN, linewidth=1.0, label="before")
        self.ax_wav.plot(a, color=MAG, linewidth=1.0, alpha=0.85, label="after")
        self.ax_wav.set_xlim(0, len(b) - 1)
        ylo = float(min(b.min(), a.min()))
        yhi = float(max(b.max(), a.max()))
        pad = 0.05 * max(1e-6, (yhi - ylo))
        self.ax_wav.set_ylim(ylo - pad, yhi + pad)
        self.ax_wav.set_title("WAVEFORM — BEFORE vs AFTER", color=MAG, fontweight="bold")
        self.ax_wav.set_xlabel("Samples", color=MUTED)
        self.ax_wav.set_ylabel("Amplitude", color=MUTED)
        leg2 = self.ax_wav.legend(frameon=True)
        leg2.get_frame().set_facecolor(PANEL)
        leg2.get_frame().set_edgecolor(GRID)

        low_before = band_db_power(b, SAMPLING_RATE, 0.0, hp)
        low_after = band_db_power(a, SAMPLING_RATE, 0.0, hp)
        high_before = band_db_power(b, SAMPLING_RATE, lp_eff, nyq)
        high_after = band_db_power(a, SAMPLING_RATE, lp_eff, nyq)

        r0 = rms_db(self.before)
        r1 = rms_db(self.after)
        p0 = peak_abs(self.before)
        p1 = peak_abs(self.after)

        self.meta.configure(text=f"sample={self.sample_idx} | window={self.window_idx}/{len(self.windows)-1} | windows={len(self.windows)} | label={self.label}")
        self.cred.configure(
            text=
            f"RMS(dB): {r0:.1f} → {r1:.1f}\n"
            f"Peak: {p0:.2f} → {p1:.2f}\n"
            f"Low(0–{int(hp)}Hz) dB power: {low_before:.1f} → {low_after:.1f}\n"
            f"High({int(lp_eff)}–Nyq) dB power: {high_before:.1f} → {high_after:.1f}\n"
            f"Expected: low/high drop; RMS may rise (gain)"
        )

        self.canvas.draw_idle()

    def _on_hover(self, event):
        if event.inaxes not in (self.ax_sb, self.ax_sa):
            return
        if event.xdata is None or event.ydata is None:
            return

        is_before = (event.inaxes == self.ax_sb)
        spec = self._to_spec_db(self.before if is_before else self.after).cpu().numpy()

        dur = float(self.dur)
        t = float(event.xdata)
        y = float(event.ydata)

        t = max(0.0, min(t, dur))
        y = max(0.0, min(y, spec.shape[0] - 1))

        col = int(round((t / dur) * (spec.shape[1] - 1))) if spec.shape[1] > 1 else 0
        row = int(round(y))

        val = float(spec[row, col])
        which = "BEFORE" if is_before else "AFTER"
        self.hover.configure(text=f"{which}\ntime={t:.3f}s\nmel_bin={row}\ndB={val:.1f}")

    def _prev_sample(self):
        self._load_sample(self.sample_idx - 1)

    def _next_sample(self):
        self._load_sample(self.sample_idx + 1)

    def _prev_window(self):
        self.window_idx = int(max(0, self.window_idx - 1))
        self.win_slider.set(self.window_idx)
        self._render()

    def _next_window(self):
        self.window_idx = int(min(len(self.windows) - 1, self.window_idx + 1))
        self.win_slider.set(self.window_idx)
        self._render()

    def _go_sample(self):
        s = self.sample_entry.get().strip()
        if s == "":
            return
        try:
            idx = int(s)
        except ValueError:
            return
        self._load_sample(idx)

    def _slider_window(self, _):
        if len(self.windows) == 0:
            return
        self.window_idx = int(round(float(self.win_slider.get())))
        self._render()

def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except:
        pass
    app = SpecGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
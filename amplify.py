import torch
import torchaudio.functional as F

def amplify(window, sample_rate, train=True, gain=2.0, cutoff_high=50.0, cutoff_low=8000.0):
    if not train:
        return window
    
    window = torch.as_tensor(window, dtype=torch.float32)

    if window.ndim == 1:
        window = window.unsqueeze(0)
        
    window *= gain

    window = F.highpass_biquad(window, int(sample_rate), float(cutoff_high))

    mls_repetition_freq = float(sample_rate) / 2.0 - 1.0
    window = F.lowpass_biquad(window, int(sample_rate), float(min(cutoff_low, mls_repetition_freq)))

    max_range = window.abs().max().clamp(min=1e-8)
    if max_range > 1.0:
        window = window / max_range

    # range
    window = torch.clamp(window, -1.0, 1.0)
    return window.squeeze(0)

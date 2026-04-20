import os
import math
import wave
from datetime import datetime

import sounddevice as sd
import numpy as np


# ====================A=====
# CONFIGURATION
# =========================
OUTPUT_DIR = "/home/luke_gut/Drone-Detection/test"
TOTAL_DURATION = 10          # total recording time in seconds
CLIP_DURATION = 5            # each clip length in seconds
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Set to None to use default input device.
# Or set to a specific device index after checking list_input_devices()
INPUT_DEVICE = None


def list_input_devices():
    """Print all available input-capable devices."""
    devices = sd.query_devices()
    print("\nAvailable audio input devices:")
    found = False
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            found = True
            print(f"[{i}] {dev['name']} | inputs={dev['max_input_channels']} | default_sr={dev['default_samplerate']}")
    if not found:
        print("No input devices found.")
    print()


def get_input_device():
    """Return a usable input device index or raise a helpful error."""
    if INPUT_DEVICE is not None:
        info = sd.query_devices(INPUT_DEVICE, "input")
        return INPUT_DEVICE

    try:
        default_input, default_output = sd.default.device
        if default_input is not None and default_input >= 0:
            sd.query_devices(default_input, "input")
            return default_input
    except Exception:
        pass

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            return i

    raise RuntimeError(
        "No usable input microphone found. Run list_input_devices() and choose one, "
        "or check whether your OS/WSL environment exposes a microphone."
    )


def save_wav(filename, audio_data, sample_rate, channels):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def record_clips(output_dir, total_duration, clip_duration, sample_rate, channels):
    os.makedirs(output_dir, exist_ok=True)

    device = get_input_device()
    device_info = sd.query_devices(device, "input")
    print(f"Using input device [{device}]: {device_info['name']}")

    num_clips = math.ceil(total_duration / clip_duration)
    print(f"Saving audio clips to: {os.path.abspath(output_dir)}")
    print(f"Total duration: {total_duration}s")
    print(f"Clip duration: {clip_duration}s")
    print(f"Number of clips: {num_clips}")
    print("Recording started...")

    for i in range(num_clips):
        remaining_time = total_duration - (i * clip_duration)
        current_clip_duration = min(clip_duration, remaining_time)
        num_samples = int(current_clip_duration * sample_rate)

        print(f"Recording clip {i + 1}/{num_clips} ({current_clip_duration}s)...")
        audio = sd.rec(
            num_samples,
            samplerate=sample_rate,
            channels=channels,
            dtype=DTYPE,
            device=device
        )
        sd.wait()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"clip_{i+1:03d}_{timestamp}.wav")
        save_wav(filename, audio, sample_rate, channels)
        print(f"Saved: {filename}")

    print("Recording complete.")


if __name__ == "__main__":
    list_input_devices()
    record_clips(
        output_dir=OUTPUT_DIR,
        total_duration=TOTAL_DURATION,
        clip_duration=CLIP_DURATION,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS
    )
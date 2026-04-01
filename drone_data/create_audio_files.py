import os
import math
import wave
from datetime import datetime

import sounddevice as sd
import numpy as np

# =========================
# CONFIGURATION
# =========================
OUTPUT_DIR = r"C:\Users\Administrator\Documents\drone_data\white_lightup"
TOTAL_DURATION = 400   # total recording time in seconds
CLIP_DURATION = 5         # each saved clip length in seconds
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def save_wav(filename, audio_data, sample_rate, channels):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def choose_input_device():
    devices = sd.query_devices()
    input_devices = []

    print("\nAvailable input devices:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append(i)
            print(f"[{i}] {dev['name']} | inputs={dev['max_input_channels']} | default_sr={dev['default_samplerate']}")

    if not input_devices:
        raise RuntimeError("No microphone/input devices were found.")

    default_in, _ = sd.default.device
    if default_in is not None and default_in >= 0:
        print(f"\nUsing default input device: [{default_in}] {sd.query_devices(default_in)['name']}")
        return default_in

    chosen = input_devices[0]
    print(f"\nNo default input set. Using first available: [{chosen}] {sd.query_devices(chosen)['name']}")
    return chosen


def record_clips(output_dir, total_duration, clip_duration, sample_rate, channels):
    os.makedirs(output_dir, exist_ok=True)
    device = choose_input_device()

    num_clips = math.ceil(total_duration / clip_duration)

    print(f"\nSaving audio clips to: {os.path.abspath(output_dir)}")
    print(f"Total duration: {total_duration}s")
    print(f"Clip duration: {clip_duration}s")
    print(f"Number of clips: {num_clips}")
    print("Recording started...")

    for i in range(num_clips):
        remaining_time = total_duration - (i * clip_duration)
        current_clip_duration = min(clip_duration, remaining_time)
        num_samples = int(current_clip_duration * sample_rate)

        print(f"Recording clip {i+1}/{num_clips} ({current_clip_duration}s)...")
        audio = sd.rec(
            frames=num_samples,
            samplerate=sample_rate,
            channels=channels,
            dtype=DTYPE,
            device=device
        )
        sd.wait()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"clip_{i+1:03d}_{timestamp}.wav")
        save_wav(out_file, audio, sample_rate, channels)
        print(f"Saved: {out_file}")

    print("Recording complete.")


if __name__ == "__main__":
    record_clips(
        output_dir=OUTPUT_DIR,
        total_duration=TOTAL_DURATION,
        clip_duration=CLIP_DURATION,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS
    )
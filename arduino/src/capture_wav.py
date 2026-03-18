import sys
import time
import serial
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("captures")
OUTPUT_DIR.mkdir(exist_ok=True)

def next_filename():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"clip_{stamp}.wav"

def read_until_wavsize(ser, timeout=None):
    """
    Wait until we receive a line starting with 'WAVSIZE '.
    If timeout is None, wait forever.
    """
    start = time.time()

    while True:
        if timeout is not None and (time.time() - start > timeout):
            return None

        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("RX:", line)

        if line.startswith("WAVSIZE "):
            return int(line.split()[1])

def stream_to_file(ser, total_bytes, filename):
    """
    Read exactly total_bytes from serial and write to file.
    """
    remaining = total_bytes
    with open(filename, "wb") as f:
        while remaining > 0:
            chunk = ser.read(min(4096, remaining))
            if not chunk:
                raise RuntimeError("Timed out while receiving WAV data.")
            f.write(chunk)
            remaining -= len(chunk)

    print(f"Saved {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python capture_loop.py <PORT> [BAUD]")
        print("Example: python capture_loop.py COM4 921600")
        return

    port = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) >= 3 else 921600

    print(f"Opening {port} at {baud} baud...")
    ser = serial.Serial(port, baudrate=baud, timeout=5)

    print("Waiting for ESP32 to boot...")
    time.sleep(3)

    ser.reset_input_buffer()

    print("Listening continuously. Press Ctrl+C to stop.")

    try:
        while True:
            total_audio_bytes = read_until_wavsize(ser, timeout=None)

            if total_audio_bytes is None:
                continue

            filename = next_filename()

            print(f"Expecting {total_audio_bytes} bytes of audio data...")
            print(f"Reading full WAV: {total_audio_bytes + 44} bytes")
            stream_to_file(ser, total_audio_bytes + 44, filename)

    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
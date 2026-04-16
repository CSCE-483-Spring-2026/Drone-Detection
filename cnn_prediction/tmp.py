import sys
import time
import socket
from pathlib import Path
from datetime import datetime
import numpy as np
from cnn_inference_server import StatusWindow, InferenceEngine, extract_features_and_window

OUTPUT_DIR = Path("captures")
OUTPUT_DIR.mkdir(exist_ok=True)

COOLDOWN = 3.0

def next_filename():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"clip_{stamp}.wav"

# =========================
# SOCKET HELPERS
# =========================

def read_line(sock):
    """
    Equivalent to ser.readline()
    Reads until '\n'
    """
    line = b""
    while not line.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            return None
        line += chunk
    return line.decode(errors="ignore").strip()

def read_until_wavsize(sock, timeout=None):
    """
    Wait until we receive a line starting with 'WAVSIZE '.
    If timeout is None, wait forever.
    """
    start = time.time()

    while True:
        if timeout is not None and (time.time() - start > timeout):
            return None

        line = read_line(sock)

        if line is None:
            return None

        if line:
            print("RX:", line)

        if line.startswith("WAVSIZE "):
            return int(line.split()[1])

def stream_to_file(sock, total_bytes, filename):
    """
    Read exactly total_bytes from socket and write to file.
    """
    remaining = total_bytes

    with open(filename, "wb") as f:
        while remaining > 0:
            chunk = sock.recv(min(4096, remaining))
            if not chunk:
                raise RuntimeError("Connection lost while receiving WAV data.")

            f.write(chunk)
            remaining -= len(chunk)

    print(f"Saved {filename}")

# =========================
# CONNECTION
# =========================

def connect(ip, port):
    while True:
        try:
            print(f"Connecting to {ip}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            print("Connected.")
            return sock
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Retrying in 2 seconds...")
            time.sleep(2)

# =========================
# MAIN
# =========================

def main():
    if len(sys.argv) < 3:
        print("Usage: python capture_loop_socket.py <IP> <PORT>")
        print("Example: python capture_loop_socket.py 192.168.1.100 12345")
        return

    ip = sys.argv[1]
    port = int(sys.argv[2])
    
    engine = InferenceEngine(detailed_logging=True)
    status_window = StatusWindow()
    cooldown_until = 0.0

    status_window.root.after(0, lambda: status_window.set_resting_state("Nothing detected. Waiting for audio..."))

    sock = connect(ip, port)

    print("Listening continuously. Press Ctrl+C to stop.")

    try:
        def process_loop():
            nonlocal sock, cooldown_until

            while True:
                total_audio_bytes = read_until_wavsize(sock, timeout=None)

                if total_audio_bytes is None:
                    print("Connection lost. Reconnecting...")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = connect(ip, port)
                    continue

                now = time.time()
                filename = next_filename()

                print(f"Expecting {total_audio_bytes} bytes of audio data...")
                print(f"Reading full WAV: {total_audio_bytes + 44} bytes")

                try:
                    stream_to_file(sock, total_audio_bytes + 44, filename)
                except RuntimeError as e:
                    print(e)
                    print("Reconnecting...")
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = connect(ip, port)

                    # Remove partial file if it exists
                    try:
                        if filename.exists():
                            filename.unlink()
                    except Exception:
                        pass

                    continue

                if now < cooldown_until:
                    print("In cooldown, skipping inference.")
                    continue

                try:
                    status_window.root.after(0, status_window.start_flashing_yellow)

                    waveform = engine.load_waveform(filename)
                    features = extract_features_and_window(waveform)

                    window_preds, window_probs = engine.predict(features)

                    avg_probs = window_probs.mean(axis=0)
                    final_pred = int(np.argmax(avg_probs))
                    final_confidence = float(avg_probs[final_pred])

                    print(f"Window predictions: {window_preds}")
                    print(f"Average probabilities: {avg_probs}")
                    print(
                        f"Final prediction: {final_pred} "
                        f"with confidence {final_confidence:.4f}"
                    )

                    status_window.root.after(
                        0,
                        lambda p=final_pred: status_window.set_result(p, hold_ms=3000)
                    )

                    cooldown_until = time.time() + COOLDOWN

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    status_window.root.after(
                        0,
                        lambda: status_window.set_resting_state(
                            "Nothing detected. Waiting for audio..."
                        )
                    )

        import threading
        worker = threading.Thread(target=process_loop, daemon=True)
        worker.start()

        status_window.run()

    except KeyboardInterrupt:
        print("\nStopped by user.")
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
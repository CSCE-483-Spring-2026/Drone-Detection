import argparse
import time
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt


def open_serial(port: str, baud: int) -> serial.Serial:
    # timeout keeps readline() from blocking forever
    return serial.Serial(port=port, baudrate=baud, timeout=1)


def parse_int_line(line: bytes):
    """Return int from a line like b'-1234\\r\\n' or None if not parseable."""
    try:
        s = line.decode("utf-8", errors="ignore").strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Live plot Arduino serial audio samples.")
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM6 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    ap.add_argument("--window", type=float, default=2.0, help="Seconds of data to show")
    ap.add_argument("--fs", type=float, default=16000.0, help="Sample rate used by Arduino (for x-axis)")
    ap.add_argument("--max-fps", type=float, default=30.0, help="Limit plot refresh rate")
    args = ap.parse_args()

    # How many samples to keep in the rolling window
    n_keep = max(50, int(args.window * args.fs))
    y = deque([0] * n_keep, maxlen=n_keep)

    # For x-axis in seconds (approx; assumes uniform sampling)
    x = np.linspace(-args.window, 0, n_keep)

    ser = open_serial(args.port, args.baud)
    # Give Arduino time to reset when serial opens (common on many boards)
    time.sleep(2.0)
    ser.reset_input_buffer()

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(x, list(y))
    ax.set_title("Live Audio Samples from Arduino (Serial)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sample (int16-ish)")
    ax.set_xlim(-args.window, 0)

    # Start with a reasonable y-range; auto-adjust as needed
    ax.set_ylim(-10000, 10000)
    fig.tight_layout()

    last_draw = 0.0
    min_dt = 1.0 / max(1.0, args.max_fps)

    try:
        while True:
            raw = ser.readline()
            if not raw:
                # nothing arrived this cycle
                continue

            v = parse_int_line(raw)
            if v is None:
                continue

            y.append(v)

            # Rate-limit plot refresh
            now = time.time()
            if now - last_draw >= min_dt:
                last_draw = now
                line.set_ydata(list(y))

                # Optional: auto-scale y a bit (comment out if you want fixed scale)
                ymax = max(2000, int(max(abs(min(y)), abs(max(y))) * 1.2))
                ax.set_ylim(-ymax, ymax)

                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
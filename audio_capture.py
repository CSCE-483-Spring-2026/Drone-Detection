import subprocess

# --- WSLg mic capture via PulseAudio (RDPSource) ---
def record_audio(filename: str, duration: float, sample_rate: int = 16000, source: str = "RDPSource") -> str:
    """
    Records audio inside WSLg by pulling from the PulseAudio source (your Windows mic).
    Requires: ffmpeg + pulseaudio-utils
      sudo apt install -y ffmpeg pulseaudio-utils
    """
    # (Optional) verify the source exists; if not, this will throw and you’ll see why.
    # You can comment this out once stable.
    try:
        out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True)
        if source not in out:
            # fallback: try to auto-pick a source containing "RDPSource"
            for line in out.splitlines():
                if "RDPSource" in line:
                    source = line.split()[1]
                    break
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "pulse",
        "-i", source,
        "-t", str(duration),
        "-ar", str(sample_rate),
        "-ac", "1",
        filename,
    ]
    # Capture stderr so you get a useful error if it fails
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg recording failed:\n{proc.stderr}")

    return filename

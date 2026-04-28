import socket
import wave
import io
import numpy as np
import threading
import time

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005

def make_wav(freq):
    t = np.linspace(0, 3, 16000 * 3, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()

def send(node_id, freq):
    data = make_wav(freq)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, SERVER_PORT))
        s.sendall(f"{node_id}\n{len(data)}\n\n".encode())
        s.sendall(data)
        print(s.recv(16).decode().strip())

t1 = threading.Thread(target=send, args=("node_1", 300))
t2 = threading.Thread(target=send, args=("node_2", 800))
t1.start()
time.sleep(0.3)
t2.start()
t1.join()
t2.join()

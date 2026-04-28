import socket
import threading
import os

HOST = "0.0.0.0"
PORT = 5005
os.makedirs("node_recordings", exist_ok=True)

def handle_node(conn):
    try:
        header = b""
        while b"\n\n" not in header:
            header += conn.recv(1)
        parts = header.decode().strip().split("\n")
        node_id = parts[0].strip()
        wav_size = int(parts[1].strip())

        buf = b""
        while len(buf) < wav_size:
            buf += conn.recv(wav_size - len(buf))

        with open(f"node_recordings/{node_id}.wav", "wb") as f:
            f.write(buf)

        print(f"{node_id} saved")
        conn.sendall(b"ACK\n")
    except Exception as e:
        print(e)
    finally:
        conn.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(10)
print(f"listening on {PORT}")

while True:
    conn, addr = server.accept()
    threading.Thread(target=handle_node, args=(conn,), daemon=True).start()

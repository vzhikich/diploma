import socket

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5005        # must match the port in the Pi script

def main():
    print(f"Listening on {HOST}:{PORT} ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        while True:
            conn, addr = s.accept()
            print("Connected by", addr)
            with conn:
                buffer = b""
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("Client disconnected")
                        break
                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        sign = line.decode("utf-8", errors="ignore").strip()
                        if sign:
                            print("Received sign:", sign)

if __name__ == "__main__":
    main()
import socket

HOST = "your.pi.hostname.or.ip"   # e.g. "raspberrypi.local" or public DNS/IP
PORT = 5005                       # must match SERVER_PORT on Pi

def main():
    print(f"Connecting to {HOST}:{PORT} ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected!")

        buffer = b""
        while True:
            data = s.recv(1024)
            if not data:
                print("Connection closed by server")
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                sign = line.decode("utf-8", errors="ignore").strip()
                if sign:
                    print("Sign:", sign)

if __name__ == "__main__":
    main()

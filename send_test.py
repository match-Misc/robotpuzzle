import socket

HOST = "0.0.0.0"  # listen on all network interfaces
PORT = 30020  # same port the robot connects to

# Predefined responses
POINTS = {
    "Point_1": "(0.4, 0, 0.5, 0, -3.14159, 0)",
    "Point_2": "(0.3, 0.5, 0.5, 0, 3.14159, 0)",
    "Point_3": "(0, 0.6, 0.5, 0, 3.14159, 0)",
}

print(f"Opening server on {HOST}:{PORT}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("Waiting for robot connection...")

    while True:
        conn, addr = server.accept()
        print(f"Robot connected from {addr}")

        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Robot closed the connection.")
                    break

                # Decode ASCII message
                msg = data.decode("ascii").strip()
                print(f"Received from robot: {msg}")

                # Choose a reply based on the message
                if msg == "2":
                    print(f"Sending response: {POINTS['Point_2']}")
                    conn.sendall((POINTS["Point_2"] + "\n").encode("ascii"))
                else:
                    print("Unknown request, sending nothing.")

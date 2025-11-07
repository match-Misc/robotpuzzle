import json
import math
import socket
import threading
import time

HOST = "0.0.0.0"  # listen on all network interfaces
PORT = 30020  # same port the robot connects to


class PoseSender:
    def __init__(self):
        self.server_socket = None
        self.client_conn = None
        self.client_addr = None
        self.is_running = False
        self.ready = False  # flag to indicate robot sent "ok"
        self.current_poses = []
        self.points = {}
        self.on_pose_sent = None  # callback when pose is sent
        self.on_message_received = None  # callback when message received from robot
        self.on_message_sent = None  # callback when message sent to robot
        self.on_robot_ready = None  # callback when robot sends "ok"

    def start_server(self):
        """Start the pose server in a separate thread"""
        if self.is_running:
            return

        self.is_running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

    def stop_server(self):
        """Stop the pose server"""
        self.is_running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except Exception:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

    def _run_server(self):
        """Main server loop"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen(1)
            print(f"Pose server listening on {HOST}:{PORT}")

            while self.is_running:
                try:
                    self.server_socket.settimeout(1.0)  # Timeout to check is_running
                    self.client_conn, self.client_addr = self.server_socket.accept()
                    print(f"Robot connected from {self.client_addr}")

                    self._handle_client()
                except socket.timeout:
                    continue
                except OSError:
                    break  # Socket was closed

        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.is_running = False

    def _handle_client(self):
        """Handle communication with connected robot"""
        try:
            with self.client_conn:
                while self.is_running:
                    data = self.client_conn.recv(1024)
                    if not data:
                        print("Robot disconnected")
                        break

                    msg = data.decode("ascii").strip()
                    print(f"Received from robot: {msg}")

                    # Callback for received message
                    if self.on_message_received:
                        self.on_message_received(msg)

                    if msg == "ok":
                        self.ready = True
                        print("Robot is ready")
                        if self.on_robot_ready:
                            self.on_robot_ready()
                    elif msg == "placed":
                        # Short timeout before sending next pose
                        time.sleep(0.5)
                        # Send next pose pair after receiving "placed"
                        if self.current_poses and self.current_pose_index < len(
                            self.current_poses
                        ):
                            self._send_next_pose_pair()
                        else:
                            print("All poses sent or no poses loaded.")
                    elif msg in self.points:
                        print(f"Sending response: {self.points[msg]}")
                        self.client_conn.sendall(
                            (self.points[msg] + "\n").encode("ascii")
                        )
                        if self.on_message_sent:
                            self.on_message_sent(self.points[msg])
                    else:
                        print(f"Unknown message: {msg}")

        except Exception as e:
            print(f"Client handling error: {e}")

    def _send_next_pose_pair(self):
        """Send the next pose pair (pickup and target) for the current piece"""
        if self.current_pose_index < len(self.current_poses):
            pose = self.current_poses[self.current_pose_index]
            # Apply offsets to target coordinates if available, to avoid collisions
            pickup_x = pose.get("pickup_x", 0)
            pickup_y = pose.get("pickup_y", 0)
            target_x = pose.get("target_x", 0) + pose.get("offset_x", 0)
            target_y = pose.get("target_y", 0) + pose.get("offset_y", 0)
            # Create single string with pickup and target coordinates
            rotation_rad = math.radians(pose["rotation"])
            print(f"Rotation (rad): {rotation_rad}")
            rotation_deg = pose["rotation"]
            pose_str = f"({pickup_x:.6f}, {pickup_y:.6f}, 0.0, {target_x:.6f}, {target_y:.6f}, {rotation_deg:.6f})"

            try:
                self.client_conn.sendall((pose_str + "\n").encode("ascii"))
                print(
                    f"Sent pose {self.current_pose_index + 1}/{len(self.current_poses)}: {pose_str}"
                )

                # Callback for sent message
                if self.on_message_sent:
                    self.on_message_sent(pose_str)

                if self.on_pose_sent:
                    self.on_pose_sent(self.current_pose_index)

                self.current_pose_index += 1

            except Exception as e:
                print(f"Error sending pose: {e}")
        else:
            print("No more poses to send")

    def load_poses_from_json(self, json_file, num_pieces=None):
        """Load poses from JSON file, optionally limiting to first N pieces"""
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            self.current_poses = data
            if num_pieces and num_pieces < len(data):
                self.current_poses = data[:num_pieces]

            self.current_pose_index = 0
            print(f"Loaded {len(self.current_poses)} poses from {json_file}")
            return True

        except Exception as e:
            print(f"Error loading poses: {e}")
            return False

    def set_points(self, points_dict):
        """Set predefined points dictionary"""
        self.points = points_dict

    def start_sending_poses(self):
        """Start sending poses to robot"""
        if not self.client_conn:
            print("No robot connected")
            return False

        if not self.ready:
            print("Robot not ready yet")
            return False

        self.current_pose_index = 0
        self._send_next_pose_pair()
        return True


# Global pose sender instance
pose_sender = PoseSender()


def start_pose_server():
    """Start the pose server (for backward compatibility)"""
    pose_sender.start_server()


def stop_pose_server():
    """Stop the pose server"""
    pose_sender.stop_server()


def load_poses(json_file, num_pieces=None):
    """Load poses from JSON file"""
    return pose_sender.load_poses_from_json(json_file, num_pieces)


def set_points(points_dict):
    """Set predefined points dictionary"""
    pose_sender.set_points(points_dict)


def send_poses():
    """Start sending poses"""
    return pose_sender.start_sending_poses()


# Backward compatibility - if run directly, start server
if __name__ == "__main__":
    pose_sender.start_server()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pose_sender.stop_server()

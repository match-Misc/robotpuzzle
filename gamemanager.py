import json
import threading
import time

import customtkinter as ctk
import requests

from nfc_reader import assign_player_name_to_server, scan_nfc_once
from send_pose import load_poses, pose_sender, start_pose_server, stop_pose_server

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Server configuration
SERVER_URL = "http://10.145.8.50:5000"
PUZZLE_ENDPOINT = f"{SERVER_URL}/api/puzzle"


class GameManagerGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Puzzle Game Manager")
        self.root.geometry("800x600")

        # Game state variables
        self.nfc_id = None
        self.player_name = None
        self.difficulty = None
        self.game_active = False
        self.countdown_active = False
        self.stopwatch_active = False
        self.start_time = None
        self.elapsed_time = 0.0
        self.poses_sent = 0
        self.total_poses = 0

        # Create GUI elements
        self.create_widgets()

        # Start pose server
        start_pose_server()
        pose_sender.on_pose_sent = self.on_pose_sent
        pose_sender.on_piece_placed = self.on_piece_placed
        pose_sender.on_all_poses_sent = self.on_all_poses_sent
        pose_sender.on_message_received = self.on_message_received
        pose_sender.on_message_sent = self.on_message_sent
        pose_sender.on_robot_ready = self.on_robot_ready

    def create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Puzzle Game Manager",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.pack(pady=(20, 30))

        # NFC Scan Section
        nfc_frame = ctk.CTkFrame(main_frame)
        nfc_frame.pack(fill="x", padx=20, pady=(0, 20))

        nfc_title = ctk.CTkLabel(
            nfc_frame, text="Step 1: NFC Scan", font=ctk.CTkFont(size=16, weight="bold")
        )
        nfc_title.pack(pady=(10, 5))

        self.scan_btn = ctk.CTkButton(
            nfc_frame,
            text="Scan NFC Tag",
            command=self.scan_nfc,
            fg_color="blue",
            height=40,
        )
        self.scan_btn.pack(pady=(0, 10))

        self.nfc_status_label = ctk.CTkLabel(
            nfc_frame, text="Ready to scan", text_color="gray"
        )
        self.nfc_status_label.pack(pady=(0, 10))

        # Name Input Section
        name_frame = ctk.CTkFrame(main_frame)
        name_frame.pack(fill="x", padx=20, pady=(0, 20))

        name_title = ctk.CTkLabel(
            name_frame,
            text="Step 2: Player Name",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        name_title.pack(pady=(10, 5))

        self.name_entry = ctk.CTkEntry(
            name_frame, placeholder_text="Enter player name", height=35
        )
        self.name_entry.pack(pady=(0, 5), fill="x", padx=10)

        self.save_name_btn = ctk.CTkButton(
            name_frame,
            text="Save Name",
            command=self.save_player_name,
            fg_color="green",
            height=35,
        )
        self.save_name_btn.pack(pady=(0, 10))

        # Difficulty Selection Section
        difficulty_frame = ctk.CTkFrame(main_frame)
        difficulty_frame.pack(fill="x", padx=20, pady=(0, 20))

        difficulty_title = ctk.CTkLabel(
            difficulty_frame,
            text="Step 3: Select Difficulty",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        difficulty_title.pack(pady=(10, 5))

        self.difficulty_var = ctk.StringVar(value="")
        difficulties = ["leicht", "mittel", "schwer"]

        for diff in difficulties:
            radio = ctk.CTkRadioButton(
                difficulty_frame,
                text=diff.capitalize(),
                variable=self.difficulty_var,
                value=diff,
                command=self.on_difficulty_select,
            )
            radio.pack(pady=2)

        # Game Control Section
        game_frame = ctk.CTkFrame(main_frame)
        game_frame.pack(fill="x", padx=20, pady=(0, 20))

        game_title = ctk.CTkLabel(
            game_frame,
            text="Step 4: Game Control",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        game_title.pack(pady=(10, 5))

        # Timer display
        self.timer_label = ctk.CTkLabel(
            game_frame,
            text="00:00.00",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color="blue",
        )
        self.timer_label.pack(pady=(10, 20))

        # Start game button
        self.start_game_btn = ctk.CTkButton(
            game_frame,
            text="Start Game",
            command=self.start_game,
            fg_color="green",
            height=45,
            state="disabled",
        )
        self.start_game_btn.pack(pady=(0, 10))

        # Stop game button
        self.stop_game_btn = ctk.CTkButton(
            game_frame,
            text="Stop Game",
            command=self.stop_game,
            fg_color="red",
            height=45,
            state="disabled",
        )
        self.stop_game_btn.pack(pady=(0, 10))

        # Status label
        self.game_status_label = ctk.CTkLabel(
            game_frame, text="Waiting for player setup", text_color="gray"
        )
        self.game_status_label.pack(pady=(10, 0))

        # Winner Selection Section
        winner_frame = ctk.CTkFrame(main_frame)
        winner_frame.pack(fill="x", padx=20, pady=(0, 20))

        winner_title = ctk.CTkLabel(
            winner_frame,
            text="Step 5: Winner Selection",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        winner_title.pack(pady=(10, 5))

        winner_buttons_frame = ctk.CTkFrame(winner_frame, fg_color="transparent")
        winner_buttons_frame.pack(pady=10)

        self.winner_btn = ctk.CTkButton(
            winner_buttons_frame,
            text="Spieler*in hat gewonnen",
            command=lambda: self.select_winner(True),
            fg_color="gold",
            height=40,
            state="disabled",
        )
        self.winner_btn.pack(side="left", padx=(0, 10))

        self.loser_btn = ctk.CTkButton(
            winner_buttons_frame,
            text="Roboter hat gewonnen",
            command=lambda: self.select_winner(False),
            fg_color="orange",
            height=40,
            state="disabled",
        )
        self.loser_btn.pack(side="left", padx=0)

        # Progress info
        self.progress_label = ctk.CTkLabel(
            winner_frame, text="Pieces placed: 0/0", text_color="gray"
        )
        self.progress_label.pack(pady=(5, 10))

    def scan_nfc(self):
        """Scan NFC tag"""
        self.scan_btn.configure(state="disabled", text="Scanning...")
        self.nfc_status_label.configure(
            text="Scanning for NFC tag...", text_color="blue"
        )

        # Run NFC scan in thread
        threading.Thread(target=self._scan_nfc_thread, daemon=True).start()

    def _scan_nfc_thread(self):
        """NFC scanning thread"""
        nfc_id, player_info = scan_nfc_once(timeout=10)

        self.root.after(0, lambda: self._handle_nfc_result(nfc_id, player_info))

    def _handle_nfc_result(self, nfc_id, player_info):
        """Handle NFC scan result"""
        self.scan_btn.configure(state="normal", text="Scan NFC Tag")

        if nfc_id and player_info:
            self.nfc_id = nfc_id
            exists = player_info.get("exists", False)
            has_name = player_info.get("has_name", False)
            name = player_info.get("player_name", "Unbenannt")

            if exists and has_name and name != "Unbenannt":
                self.player_name = name
                self.nfc_status_label.configure(
                    text=f"Player: {name} (ID: {nfc_id})", text_color="green"
                )
                self.name_entry.delete(0, "end")
                self.name_entry.insert(0, name)
                self.save_name_btn.configure(state="normal")
            else:
                self.nfc_status_label.configure(
                    text=f"NFC found but needs name (ID: {nfc_id})", text_color="orange"
                )
                self.name_entry.delete(0, "end")
                self.save_name_btn.configure(state="disabled")
        else:
            self.nfc_status_label.configure(
                text="No NFC tag detected", text_color="red"
            )

    def save_player_name(self):
        """Save player name to server"""
        if not self.nfc_id:
            return

        name = self.name_entry.get().strip()
        if not name:
            return

        self.save_name_btn.configure(state="disabled", text="Saving...")

        # Save name in thread
        threading.Thread(
            target=self._save_name_thread, args=(name,), daemon=True
        ).start()

    def _save_name_thread(self, name):
        """Save name thread"""
        success = assign_player_name_to_server(self.nfc_id, name)

        self.root.after(0, lambda: self._handle_name_save(success, name))

    def _handle_name_save(self, success, name):
        """Handle name save result"""
        self.save_name_btn.configure(state="normal", text="Save Name")

        if success:
            self.player_name = name
            self.nfc_status_label.configure(
                text=f"Player: {name} (ID: {self.nfc_id})", text_color="green"
            )
        else:
            self.nfc_status_label.configure(
                text="Failed to save name", text_color="red"
            )

    def on_difficulty_select(self):
        """Handle difficulty selection"""
        self.difficulty = self.difficulty_var.get()
        if self.difficulty:
            self.start_game_btn.configure(state="normal")
        else:
            self.start_game_btn.configure(state="disabled")

    def start_game(self):
        """Start the game"""
        if not self.difficulty:
            return

        # Determine number of pieces based on difficulty
        if self.difficulty == "leicht":
            num_pieces = 12
        else:  # mittel or schwer
            num_pieces = 24

        # Load poses
        json_file = "solved_puzzle_24_with_offsets.json"
        if load_poses(json_file, num_pieces):
            self.total_poses = num_pieces
            self.poses_sent = 0
            self.update_progress()

            # Start countdown
            self.start_countdown()
        else:
            self.game_status_label.configure(
                text="Failed to load puzzle data", text_color="red"
            )

    def start_countdown(self):
        """Start countdown from 10"""
        self.countdown_active = True
        self.countdown_value = 10
        self.timer_label.configure(text_color="orange")
        self.game_status_label.configure(text="Get ready!", text_color="orange")
        self.start_game_btn.configure(state="disabled")

        # Start countdown thread
        threading.Thread(target=self._countdown_thread, daemon=True).start()

    def _countdown_thread(self):
        """Countdown thread"""
        while self.countdown_active and self.countdown_value > 0:
            self.root.after(
                0,
                lambda v=self.countdown_value: self.timer_label.configure(
                    text=f"{v:02d}"
                ),
            )
            time.sleep(1)
            self.countdown_value -= 1

        if self.countdown_active:  # Countdown completed
            self.root.after(0, self.start_stopwatch)

    def start_stopwatch(self):
        """Start the stopwatch"""
        self.countdown_active = False
        self.stopwatch_active = True
        self.game_active = True
        self.start_time = time.time()
        self.elapsed_time = 0.0

        self.timer_label.configure(text_color="green")
        self.game_status_label.configure(text="Game running!", text_color="green")
        self.stop_game_btn.configure(state="normal")

        # Start stopwatch thread
        threading.Thread(target=self._stopwatch_thread, daemon=True).start()

        # Start sending poses after countdown finishes
        pose_sender.start_sending_poses()

    def _stopwatch_thread(self):
        """Stopwatch thread"""
        while self.stopwatch_active:
            if self.start_time:
                self.elapsed_time = time.time() - self.start_time
                minutes = int(self.elapsed_time // 60)
                seconds = int(self.elapsed_time % 60)
                centiseconds = int((self.elapsed_time % 1) * 100)
                time_str = f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
                self.root.after(
                    0, lambda t=time_str: self.timer_label.configure(text=t)
                )
            time.sleep(0.01)

    def stop_game(self):
        """Stop the game"""
        self.game_active = False
        self.stopwatch_active = False
        self.countdown_active = False

        self.timer_label.configure(text_color="red")
        self.game_status_label.configure(text="Game stopped", text_color="red")
        self.stop_game_btn.configure(state="disabled")

        # Enable winner selection
        self.winner_btn.configure(state="normal")
        self.loser_btn.configure(state="normal")

    def on_pose_sent(self, pose_index):
        """Callback when a pose is sent"""
        self.poses_sent = pose_index + 1
        self.root.after(0, self.update_progress)

    def on_piece_placed(self):
        """Callback when a piece is placed"""
        pass  # Could add visual feedback here

    def on_all_poses_sent(self):
        """Callback when all poses are sent"""
        self.root.after(0, self._handle_all_poses_sent)

    def on_message_received(self, message):
        """Callback when message received from robot"""
        print(f"Received from robot: {message}")

    def on_robot_ready(self):
        """Callback when robot sends 'ok'"""
        print("Robot is ready for game")
        # Could update UI here if needed

    def on_message_sent(self, message):
        """Callback when message sent to robot"""
        print(f"Sent to robot: {message}")

    def _handle_all_poses_sent(self):
        """Handle when all poses are sent"""
        if self.game_active:
            self.game_status_label.configure(
                text="All pieces sent! Complete placement.", text_color="blue"
            )

    def update_progress(self):
        """Update progress display"""
        self.progress_label.configure(
            text=f"Pieces sent: {self.poses_sent}/{self.total_poses}"
        )

    def select_winner(self, is_winner):
        """Select winner and submit result"""
        if not self.elapsed_time:
            return

        # Disable buttons
        self.winner_btn.configure(state="disabled")
        self.loser_btn.configure(state="disabled")

        # Submit result only if nfc_id and player_name are set
        if self.nfc_id and self.player_name:
            threading.Thread(
                target=self._submit_result_thread, args=(is_winner,), daemon=True
            ).start()
        else:
            # Skip submission, just reset
            self.game_status_label.configure(
                text="Game completed (no submission)", text_color="blue"
            )
            self.reset_game()

    def _submit_result_thread(self, is_winner):
        """Submit result thread"""
        try:
            # Map difficulty to API format
            difficulty_map = {
                "leicht": "Leicht",
                "mittel": "Mittel",
                "schwer": "Schwer",
            }

            payload = {
                "nfc_id": self.nfc_id,
                "time": round(self.elapsed_time, 2),
                "difficulty": difficulty_map.get(self.difficulty, "Mittel"),
            }

            headers = {"Content-Type": "application/json"}

            response = requests.post(
                PUZZLE_ENDPOINT, json=payload, headers=headers, timeout=10
            )

            success = response.status_code == 200
            self.root.after(0, lambda s=success: self._handle_submit_result(s))

        except Exception as e:
            print(f"Submit error: {e}")
            self.root.after(0, lambda: self._handle_submit_result(False))

    def _handle_submit_result(self, success):
        """Handle submit result"""
        if success:
            self.game_status_label.configure(
                text="Result submitted successfully!", text_color="green"
            )
            # Reset for next game
            self.reset_game()
        else:
            self.game_status_label.configure(
                text="Failed to submit result", text_color="red"
            )
            # Re-enable buttons to try again
            self.winner_btn.configure(state="normal")
            self.loser_btn.configure(state="normal")

    def reset_game(self):
        """Reset game state for next player"""
        self.nfc_id = None
        self.player_name = None
        self.difficulty = None
        self.game_active = False
        self.elapsed_time = 0.0
        self.poses_sent = 0
        self.total_poses = 0

        # Reset UI
        self.nfc_status_label.configure(text="Ready to scan", text_color="gray")
        self.name_entry.delete(0, "end")
        self.difficulty_var.set("")
        self.timer_label.configure(text="00:00.00", text_color="blue")
        self.game_status_label.configure(
            text="Waiting for player setup", text_color="gray"
        )
        self.progress_label.configure(text="Pieces sent: 0/0")
        self.start_game_btn.configure(state="disabled")
        self.stop_game_btn.configure(state="disabled")
        self.winner_btn.configure(state="disabled")
        self.loser_btn.configure(state="disabled")
        self.save_name_btn.configure(state="disabled")

    def run(self):
        self.root.mainloop()

    def __del__(self):
        """Cleanup on exit"""
        stop_pose_server()


if __name__ == "__main__":
    app = GameManagerGUI()
    app.run()

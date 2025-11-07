import send_pose


def on_message_received(msg):
    """Callback for received messages - print raw data"""
    print(f"RAW RECEIVED: {repr(msg)}")


def on_message_sent(msg):
    """Callback for sent messages - print raw data"""
    print(f"RAW SENT: {repr(msg)}")


def main():
    # Load poses from the JSON file
    if not send_pose.load_poses("solved_puzzle_24_with_offsets.json"):
        print("Failed to load poses")
        return

    # Set up callbacks for raw data printing
    send_pose.pose_sender.on_message_received = on_message_received
    send_pose.pose_sender.on_message_sent = on_message_sent

    # Start the pose server
    send_pose.start_pose_server()

    # Keep the main thread alive
    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        send_pose.stop_pose_server()


if __name__ == "__main__":
    main()

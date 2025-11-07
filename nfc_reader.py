import json
import time

import requests
import serial

# --- Configuration ---
# Set the serial port your D1 Mini is connected to.
SERIAL_PORT = "COM7"

# Set the baud rate. Your README mentions 9600 for the D1 Mini.
# If this doesn't work, 115200 is another common rate.
BAUD_RATE = 9600

# Server configuration for retrieving player names
SERVER_URL = "http://10.145.8.50:5000"
NFC_SCAN_ENDPOINT = f"{SERVER_URL}/api/nfc_scan"
# ---------------------


def get_player_name_from_server(nfc_id):
    """
    Queries the server to get the player name associated with the NFC ID.

    Args:
        nfc_id (str): The NFC ID to look up

    Returns:
        dict: Server response containing player information, or None if error
    """
    try:
        payload = {"nfc_id": nfc_id}
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            NFC_SCAN_ENDPOINT, json=payload, headers=headers, timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"Server returned status code {response.status_code}: {response.text}"
            )
            return None

    except requests.exceptions.RequestException as e:
        print(f"Network error when contacting server: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing server response: {e}")
        return None


def add_nfc_chip_to_server(nfc_id):
    """
    Adds a new NFC chip to the server system.

    Args:
        nfc_id (str): The NFC ID to add

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use the admin add_nfc endpoint
        add_endpoint = f"{SERVER_URL}/admin/add_nfc"
        payload = {"nfc_id": nfc_id}

        response = requests.post(add_endpoint, data=payload, timeout=5)

        if response.status_code == 200:
            print(f"✅ Successfully added NFC ID {nfc_id} to system")
            return True
        else:
            print(f"❌ Failed to add NFC ID. Server returned: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Network error when adding NFC ID: {e}")
        return False


def assign_player_name_to_server(nfc_id, player_name):
    """
    Assigns a player name to an NFC ID on the server.

    Args:
        nfc_id (str): The NFC ID to assign name to
        player_name (str): The player name to assign

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use the admin assign_name endpoint
        assign_endpoint = f"{SERVER_URL}/admin/assign_name"
        payload = {"nfc_id": nfc_id, "name": player_name}

        response = requests.post(assign_endpoint, data=payload, timeout=5)

        if response.status_code == 200:
            print(f"✅ Successfully assigned name '{player_name}' to NFC ID {nfc_id}")
            return True
        else:
            print(f"❌ Failed to assign name. Server returned: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Network error when assigning name: {e}")
        return False


def parse_nfc_data(line_string):
    """
    Parses the NFC data line to extract the NFC ID.

    Expected formats:
    - "NFC_ID:A1B2C3D4E5F6" (Arduino format)
    - "FF0FFAA77C0100" (raw hex format)

    Args:
        line_string (str): The raw line from serial

    Returns:
        str: The NFC ID if found, None otherwise
    """
    # Check for Arduino format first
    if "NFC_ID:" in line_string:
        # Extract the NFC ID after "NFC_ID:"
        parts = line_string.split("NFC_ID:")
        if len(parts) > 1:
            nfc_id = parts[1].strip()
            return nfc_id

    # Check for raw hex format (assuming it's a valid NFC ID)
    # Basic validation: should be hexadecimal and reasonable length
    if len(line_string) >= 8 and len(line_string) <= 16:
        try:
            # Try to parse as hex
            int(line_string, 16)
            return line_string.upper()  # Normalize to uppercase
        except ValueError:
            pass

    return None


def read_from_d1_device():
    """
    Connects to the specified serial port and reads data line by line.
    Automatically handles disconnection and retries connection.
    """

    # This variable will hold the serial port object
    serial_connection = None

    while True:
        try:
            # Attempt to open the serial port
            # timeout=1 means readline() will wait 1 second for data
            # before moving on, allowing the loop to run.
            serial_connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(
                f"--- Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud ---"
            )

            # Loop forever to read data from the device
            while True:
                # Read one line of data, ending in a newline character
                line_bytes = serial_connection.readline()

                # If data was actually read (not just a timeout)
                if line_bytes:
                    try:
                        # Decode the bytes into a string (using 'utf-8')
                        # and strip any leading/trailing whitespace (like \n or \r)
                        line_string = line_bytes.decode("utf-8").strip()

                        # Print the clean data
                        print(f"Received: {line_string}")

                        # Parse NFC ID from the line
                        nfc_id = parse_nfc_data(line_string)
                        if nfc_id:
                            print(f"Detected NFC ID: {nfc_id}")

                            # Query server for player name
                            player_info = get_player_name_from_server(nfc_id)

                            if player_info:
                                if player_info.get("exists", False):
                                    player_name = player_info.get(
                                        "player_name", "Unknown"
                                    )
                                    has_name = player_info.get("has_name", False)

                                    if has_name and player_name != "Unbenannt":
                                        print(f"✅ Player found: {player_name}")
                                    else:
                                        print(
                                            f"⚠️  NFC ID exists but no name assigned: {nfc_id}"
                                        )
                                        # Prompt for player name
                                        try:
                                            new_name = input(
                                                f"Enter player name for NFC ID {nfc_id}: "
                                            ).strip()
                                            if new_name:
                                                success = assign_player_name_to_server(
                                                    nfc_id, new_name
                                                )
                                                if success:
                                                    print(
                                                        f"✅ Name '{new_name}' assigned successfully!"
                                                    )
                                                else:
                                                    print(
                                                        "❌ Failed to assign name to server"
                                                    )
                                            else:
                                                print(
                                                    "No name entered, skipping assignment"
                                                )
                                        except KeyboardInterrupt:
                                            print("\nName assignment cancelled")
                                else:
                                    print(f"❌ NFC ID not found in system: {nfc_id}")
                                    # Offer to add the NFC ID to the system
                                    try:
                                        add_choice = (
                                            input(
                                                f"Add NFC ID {nfc_id} to system? (y/n): "
                                            )
                                            .strip()
                                            .lower()
                                        )
                                        if add_choice in ["y", "yes"]:
                                            success = add_nfc_chip_to_server(nfc_id)
                                            if success:
                                                print(
                                                    "✅ NFC ID added to system. You can now assign a name."
                                                )
                                                # After adding, prompt for name
                                                try:
                                                    new_name = input(
                                                        f"Enter player name for NFC ID {nfc_id}: "
                                                    ).strip()
                                                    if new_name:
                                                        success = assign_player_name_to_server(
                                                            nfc_id, new_name
                                                        )
                                                        if success:
                                                            print(
                                                                f"✅ Name '{new_name}' assigned successfully!"
                                                            )
                                                        else:
                                                            print(
                                                                "❌ Failed to assign name to server"
                                                            )
                                                    else:
                                                        print(
                                                            "No name entered, NFC ID added without name"
                                                        )
                                                except KeyboardInterrupt:
                                                    print("\nName assignment cancelled")
                                            else:
                                                print(
                                                    "❌ Failed to add NFC ID to system"
                                                )
                                        else:
                                            print("NFC ID not added to system")
                                    except KeyboardInterrupt:
                                        print("\nNFC ID addition cancelled")
                            else:
                                print("❌ Failed to get response from server")
                        else:
                            # Not an NFC ID line, just print as before
                            pass

                    except UnicodeDecodeError:
                        # Handle cases where data isn't valid UTF-8
                        print(f"Received non-UTF-8 data: {line_bytes}")

        except serial.SerialException as e:
            # Handle errors, such as:
            # 1. Port not found (device not plugged in)
            # 2. Access denied (port already in use by Arduino IDE, etc.)
            # 3. Device disconnected while reading
            if serial_connection:
                serial_connection.close()

            print(f"\n--- Disconnected (Error: {e}) ---")
            print(
                f"Waiting for {SERIAL_PORT} to become available... (Press Ctrl+C to exit)"
            )
            time.sleep(3)  # Wait 3 seconds before trying to reconnect

        except KeyboardInterrupt:
            # Handle user pressing Ctrl+C
            print("\nExiting script.")
            if serial_connection and serial_connection.is_open:
                serial_connection.close()
            break


if __name__ == "__main__":
    read_from_d1_device()

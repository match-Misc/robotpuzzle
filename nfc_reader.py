import time

import serial

# --- Configuration ---
# Set the serial port your D1 Mini is connected to.
SERIAL_PORT = "COM11"

# Set the baud rate. Your README mentions 9600 for the D1 Mini.
# If this doesn't work, 115200 is another common rate.
BAUD_RATE = 9600
# ---------------------


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

#!/usr/bin/env python3
"""
Arduino Joystick Serial Reader
Reads joystick values from Arduino connected via USB serial port.
Expected format: /*value1,value2,...,valueN*/
    sudo usermod -a -G dialout $USER
    sudo chmod 666 /dev/ttyUSB0
"""

import serial
import time
import re


class JoystickReader:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
        """
        Initialize the joystick reader.
        
        Args:
            port: Serial port path (default: /dev/ttyUSB0)
            baudrate: Communication speed (default: 9600)
            timeout: Read timeout in seconds (default: 1)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        
        # Joystick channel mapping (0-indexed)
        self.THROTTLE = 1
        self.AILERON = 3
        self.ELEVATOR = 5
        self.RUDDER = 7
        self.SWITCH = 9
        
        # Calibration data
        self.calibration = {
            'throttle': {'min': None, 'max': None, 'name': 'Throttle'},
            'aileron': {'min': None, 'max': None, 'name': 'Aileron'},
            'elevator': {'min': None, 'max': None, 'name': 'Elevator'},
            'rudder': {'min': None, 'max': None, 'name': 'Rudder'},
            'switch': {'values': set(), 'name': 'Switch'}
        }
        self.is_calibrated = False
        
    def connect(self):
        """Establish serial connection to Arduino."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"Connected to {self.port} at {self.baudrate} baud")
            # Wait for Arduino to reset after connection
            time.sleep(2)
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def parse_data(self, raw_data):
        """
        Parse the incoming data string.
        Expected format: /*value1,value2,...,valueN*/
        
        Args:
            raw_data: Raw string from serial port
            
        Returns:
            List of integer values or None if parsing fails
        """
        # Remove comment markers and whitespace
        match = re.search(r'/\*(.+?)\*/', raw_data)
        if match:
            data_str = match.group(1)
            try:
                # Split by comma and convert to integers
                values = [int(x.strip()) for x in data_str.split(',')]
                return values
            except ValueError as e:
                print(f"Error parsing values: {e}")
                return None
        return None
    
    def calibrate(self):
        """
        Run calibration phase to determine the range of each control.
        User moves each stick through its full range.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial connection not established. Call connect() first.")
            return False
        
        print("\n" + "="*60)
        print("JOYSTICK CALIBRATION")
        print("="*60)
        print("Move each control through its FULL RANGE when prompted.")
        print("Press Enter when done with each control.\n")
        
        controls = [
            ('throttle', self.THROTTLE, 'THROTTLE stick (up/down)'),
            ('aileron', self.AILERON, 'AILERON stick (left/right)'),
            ('elevator', self.ELEVATOR, 'ELEVATOR stick (up/down)'),
            ('rudder', self.RUDDER, 'RUDDER stick (left/right)'),
        ]
        
        for control_name, channel, description in controls:
            print(f"\n>>> Move {description}")
            print(f"    Monitoring channel {channel}...")
            input("    Press Enter when ready to start...")
            
            min_val = None
            max_val = None
            start_time = time.time()
            duration = 5  # Collect data for 5 seconds
            
            print(f"    Recording for {duration} seconds...")
            while time.time() - start_time < duration:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    values = self.parse_data(line)
                    
                    if values and len(values) > channel:
                        val = values[channel]
                        
                        if min_val is None or val < min_val:
                            min_val = val
                        if max_val is None or val > max_val:
                            max_val = val
                        
                        remaining = duration - (time.time() - start_time)
                        print(f"    Current: {val:4d}  |  Range: {min_val:4d} - {max_val:4d}  |  Time left: {remaining:.1f}s", end='\r')
                
                time.sleep(0.01)
            
            self.calibration[control_name]['min'] = min_val
            self.calibration[control_name]['max'] = max_val
            print(f"\n    ✓ {description} calibrated: {min_val} - {max_val}")
        
        # Calibrate switch
        print(f"\n>>> Toggle SWITCH (channel {self.SWITCH})")
        print(f"    Flip the switch a few times...")
        
        switch_values = set()
        samples = 0
        max_samples = 50
        
        while samples < max_samples:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                values = self.parse_data(line)
                
                if values and len(values) > self.SWITCH:
                    switch_values.add(values[self.SWITCH])
                    samples += 1
                    print(f"    Switch values detected: {sorted(switch_values)}", end='\r')
            
            time.sleep(0.01)
        
        self.calibration['switch']['values'] = switch_values
        print(f"\n    ✓ Switch calibrated: {sorted(switch_values)}")
        
        # Display calibration summary
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        for control in ['throttle', 'aileron', 'elevator', 'rudder']:
            cal = self.calibration[control]
            print(f"{cal['name']:10s}: {cal['min']:4d} - {cal['max']:4d}  (range: {cal['max']-cal['min']})")
        print(f"{'Switch':10s}: {sorted(self.calibration['switch']['values'])}")
        print("="*60 + "\n")
        
        self.is_calibrated = True
        return True
    
    def get_normalized_values(self, raw_values):
        """
        Convert raw joystick values to normalized values using calibration data.
        
        Args:
            raw_values: List of raw integer values from Arduino
            
        Returns:
            Dictionary with normalized control values (-1 to 1 for sticks, 0/1 for switch)
        """
        if not self.is_calibrated:
            return None
        
        if len(raw_values) <= max(self.THROTTLE, self.AILERON, self.ELEVATOR, self.RUDDER, self.SWITCH):
            return None
        
        def normalize(value, min_val, max_val):
            """Normalize value to -1.0 to 1.0 range"""
            if max_val == min_val:
                return 0.0
            return 2.0 * (value - min_val) / (max_val - min_val) - 1.0
        
        normalized = {
            'throttle': normalize(
                raw_values[self.THROTTLE],
                self.calibration['throttle']['min'],
                self.calibration['throttle']['max']
            ),
            'aileron': normalize(
                raw_values[self.AILERON],
                self.calibration['aileron']['min'],
                self.calibration['aileron']['max']
            ),
            'elevator': normalize(
                raw_values[self.ELEVATOR],
                self.calibration['elevator']['min'],
                self.calibration['elevator']['max']
            ),
            'rudder': normalize(
                raw_values[self.RUDDER],
                self.calibration['rudder']['min'],
                self.calibration['rudder']['max']
            ),
            'switch': raw_values[self.SWITCH]
        }
        
        return normalized
    
    def read_continuous(self, callback=None):
        """
        Continuously read data from the serial port.
        
        Args:
            callback: Optional function to call with parsed values
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial connection not established. Call connect() first.")
            return
        
        print("Reading joystick data... Press Ctrl+C to stop")
        try:
            while True:
                if self.serial_conn.in_waiting > 0:
                    # Read line from serial port
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        # Parse the data
                        values = self.parse_data(line)
                        
                        if values:
                            if callback:
                                callback(values)
                            else:
                                print(f"Joystick values: {values}")
                
                time.sleep(0.01)  # Small delay to prevent CPU hogging
                
        except KeyboardInterrupt:
            print("\nStopping joystick reader...")
        except Exception as e:
            print(f"Error reading data: {e}")
    
    def close(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed")


def main():
    """Main function to demonstrate usage."""
    # Create joystick reader instance
    joystick = JoystickReader(port='/dev/ttyUSB0', baudrate=9600)
    
    # Connect to Arduino
    if joystick.connect():
        try:
            # Run calibration
            if joystick.calibrate():
                print("Starting normal operation...")
                print("Press Ctrl+C to stop\n")
                
                # Define callback to display normalized values
                def display_callback(values):
                    normalized = joystick.get_normalized_values(values)
                    if normalized:
                        print(f"T: {normalized['throttle']:+.2f}  "
                              f"A: {normalized['aileron']:+.2f}  "
                              f"E: {normalized['elevator']:+.2f}  "
                              f"R: {normalized['rudder']:+.2f}  "
                              f"SW: {normalized['switch']}", end='\r')
                
                # Read data continuously with callback
                joystick.read_continuous(callback=display_callback)
        finally:
            # Always close connection on exit
            joystick.close()
    else:
        print("Failed to connect to Arduino. Check port and permissions.")
        print("You may need to run: sudo usermod -a -G dialout $USER")
        print("Then log out and back in.")


if __name__ == "__main__":
    main()

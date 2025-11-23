#!/usr/bin/env python3
"""
Joystick Node - ROS2 wrapper for Arduino Joystick Serial Reader
Reads joystick values from Arduino and publishes them as JoystickData messages.
"""

import rclpy
from rclpy.node import Node
from my_interfaces.msg import JoystickData
import serial
import time
import re


class JoystickNode(Node):
    def __init__(self):
        super().__init__('joystick_node')
        
        # Declare parameters
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 9600)
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('calibration_duration', 5.0)  # seconds per control
        
        # Get parameters
        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.calibration_duration = self.get_parameter('calibration_duration').value
        
        # Serial connection
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
        
        # Create publisher
        self.publisher = self.create_publisher(JoystickData, 'joystick', 10)
        
        # Connect and calibrate
        if self.connect():
            if self.calibrate():
                # Create timer for publishing
                timer_period = 1.0 / self.publish_rate
                self.timer = self.create_timer(timer_period, self.timer_callback)
                self.get_logger().info('Joystick node started and publishing')
            else:
                self.get_logger().error('Calibration failed')
                self.destroy_node()
        else:
            self.get_logger().error('Failed to connect to Arduino')
            self.destroy_node()
    
    def connect(self):
        """Establish serial connection to Arduino."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.get_logger().info(f'Connected to {self.port} at {self.baudrate} baud')
            time.sleep(2)  # Wait for Arduino to reset
            return True
        except serial.SerialException as e:
            self.get_logger().error(f'Error connecting to {self.port}: {e}')
            return False
    
    def parse_data(self, raw_data):
        """
        Parse the incoming data string.
        Expected format: /*value1,value2,...,valueN*/
        """
        match = re.search(r'/\*(.+?)\*/', raw_data)
        if match:
            data_str = match.group(1)
            try:
                values = [int(x.strip()) for x in data_str.split(',')]
                return values
            except ValueError as e:
                self.get_logger().warn(f'Error parsing values: {e}')
                return None
        return None
    
    def calibrate(self):
        """Run calibration phase to determine the range of each control."""
        if not self.serial_conn or not self.serial_conn.is_open:
            self.get_logger().error('Serial connection not established')
            return False
        
        self.get_logger().info('='*60)
        self.get_logger().info('JOYSTICK CALIBRATION')
        self.get_logger().info('='*60)
        self.get_logger().info('Move each control through its FULL RANGE when prompted.')
        
        controls = [
            ('throttle', self.THROTTLE, 'THROTTLE stick (up/down)'),
            ('aileron', self.AILERON, 'AILERON stick (left/right)'),
            ('elevator', self.ELEVATOR, 'ELEVATOR stick (up/down)'),
            ('rudder', self.RUDDER, 'RUDDER stick (left/right)'),
        ]
        
        for control_name, channel, description in controls:
            self.get_logger().info(f'\n>>> Move {description}')
            self.get_logger().info(f'    Monitoring channel {channel}...')
            input('    Press Enter when ready to start...')
            
            min_val = None
            max_val = None
            start_time = time.time()
            
            self.get_logger().info(f'    Recording for {self.calibration_duration} seconds...')
            while time.time() - start_time < self.calibration_duration:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    values = self.parse_data(line)
                    
                    if values and len(values) > channel:
                        val = values[channel]
                        
                        if min_val is None or val < min_val:
                            min_val = val
                        if max_val is None or val > max_val:
                            max_val = val
                        
                        remaining = self.calibration_duration - (time.time() - start_time)
                        print(f'    Current: {val:4d}  |  Range: {min_val:4d} - {max_val:4d}  |  Time left: {remaining:.1f}s', end='\r')
                
                time.sleep(0.01)
            
            self.calibration[control_name]['min'] = min_val
            self.calibration[control_name]['max'] = max_val
            self.get_logger().info(f'\n    ✓ {description} calibrated: {min_val} - {max_val}')
        
        # Calibrate switch
        self.get_logger().info(f'\n>>> Toggle SWITCH (channel {self.SWITCH})')
        self.get_logger().info('    Flip the switch a few times...')
        
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
                    print(f'    Switch values detected: {sorted(switch_values)}', end='\r')
            
            time.sleep(0.01)
        
        self.calibration['switch']['values'] = switch_values
        self.get_logger().info(f'\n    ✓ Switch calibrated: {sorted(switch_values)}')
        
        # Display calibration summary
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('CALIBRATION COMPLETE')
        self.get_logger().info('='*60)
        for control in ['throttle', 'aileron', 'elevator', 'rudder']:
            cal = self.calibration[control]
            self.get_logger().info(f"{cal['name']:10s}: {cal['min']:4d} - {cal['max']:4d}  (range: {cal['max']-cal['min']})")
        self.get_logger().info(f"{'Switch':10s}: {sorted(self.calibration['switch']['values'])}")
        self.get_logger().info('='*60)
        
        self.is_calibrated = True
        return True
    
    def get_normalized_values(self, raw_values):
        """Convert raw joystick values to normalized values using calibration data."""
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
            'switch_state': raw_values[self.SWITCH]
        }
        
        return normalized
    
    def timer_callback(self):
        """Timer callback to read and publish joystick data."""
        if self.serial_conn and self.serial_conn.in_waiting > 0:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                raw_values = self.parse_data(line)
                
                if raw_values:
                    normalized = self.get_normalized_values(raw_values)
                    
                    if normalized:
                        msg = JoystickData()
                        msg.throttle = normalized['throttle']
                        msg.aileron = normalized['aileron']
                        msg.elevator = normalized['elevator']
                        msg.rudder = normalized['rudder']
                        msg.switch_state = normalized['switch_state']
                        
                        self.publisher.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'Error reading joystick data: {e}')
    
    def destroy_node(self):
        """Clean up resources."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('Serial connection closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = JoystickNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import signal
import threading

class DummyAgent(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.node_name = node_name
        self._shutdown_requested = False
        
        # Create a publisher to announce this agent's existence
        self.status_publisher = self.create_publisher(
            String, 
            f'{node_name}/status', 
            10
        )
        
        # Create a timer to publish status periodically
        self.timer = self.create_timer(5.0, self.publish_status)  # Every 5 seconds
        
        self.get_logger().info(f'Dummy agent {node_name} started')
        
    def publish_status(self):
        if not self._shutdown_requested:
            msg = String()
            msg.data = f'Agent {self.node_name} is active'
            try:
                self.status_publisher.publish(msg)
                self.get_logger().debug(f'{self.node_name} status published')
            except Exception as e:
                if not self._shutdown_requested:
                    self.get_logger().warn(f'Failed to publish status: {e}')
    
    def shutdown(self):
        self._shutdown_requested = True
        self.get_logger().info(f'Shutting down {self.node_name}')

def signal_handler(signum, frame, dummy_agent):
    """Handle shutdown signals gracefully"""
    if dummy_agent:
        dummy_agent.shutdown()

def main(args=None):
    if len(sys.argv) != 2:
        print("Usage: python3 dummy_agent.py <agent_name>")
        return 1
        
    agent_name = sys.argv[1]
    dummy_agent = None
    
    try:
        rclpy.init(args=args)
        dummy_agent = DummyAgent(agent_name)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, dummy_agent))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, dummy_agent))
        
        # Spin in a way that can be interrupted
        while rclpy.ok() and not dummy_agent._shutdown_requested:
            try:
                rclpy.spin_once(dummy_agent, timeout_sec=1.0)
            except KeyboardInterrupt:
                dummy_agent.shutdown()
                break
            except Exception as e:
                if not dummy_agent._shutdown_requested:
                    dummy_agent.get_logger().error(f'Error in spin: {e}')
                break
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Clean shutdown
        try:
            if dummy_agent and not dummy_agent._shutdown_requested:
                dummy_agent.shutdown()
            if dummy_agent:
                dummy_agent.destroy_node()
        except Exception as e:
            print(f"Error during node cleanup: {e}")
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors - they're common when processes are killed
            pass

if __name__ == '__main__':
    main()
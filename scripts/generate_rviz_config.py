#!/usr/bin/env python3
"""
RViz Configuration Generator for Ergodic Exploration

This script automatically generates an RViz configuration file based on the number
of active agents in the ROS2 system. It dynamically discovers agents and creates
path displays with unique colors for each agent.

Usage:
    python generate_rviz_config.py [--max-agents MAX] [--output OUTPUT_FILE] [--discover]
    
Arguments:
    --max-agents: Maximum number of agents to configure (default: 10)
    --output: Output RViz configuration file (default: auto_rviz_config.rviz)
    --discover: Attempt to discover active agents in ROS2 network (requires ROS2 environment)
    
Examples:
    python generate_rviz_config.py --max-agents 5
    python generate_rviz_config.py --discover --output my_config.rviz
"""

import argparse
import sys
import yaml
import colorsys
from pathlib import Path

# RViz configuration template
RVIZ_CONFIG_TEMPLATE = """
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /ObstacleMap1/Position1
      Splitter Ratio: 0.5
    Tree Height: 746
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
  - Class: rviz_common/Time
    Experimental: false
    Name: Time
    SyncMode: 0
    SyncSource: ""
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 5
        Y: 5
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: AgentsArray
      Namespaces:
        agents: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /agent_markers
      Value: true
    - Alpha: 0.30000001192092896
      Class: rviz_default_plugins/Map
      Color Scheme: map
      Draw Behind: true
      Enabled: true
      Name: ObstacleMap
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /obstacle_map
      Update Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /obstacle_map_updates
      Use Timestamp: false
      Value: true{path_displays}
    - Class: rviz_default_plugins/MarkerArray
      Enabled: false
      Name: ObstacleMarkers
      Namespaces:
        {{}}
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /obstacle_markers
      Value: false
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: TargetEstimations
      Namespaces:
        ground_truths: true
        target_estimates: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /target_estimate_markers
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
    - Class: rviz_default_plugins/SetInitialPose
      Covariance x: 0.25
      Covariance y: 0.25
      Covariance yaw: 0.06853891909122467
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /clicked_point
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 13.844531059265137
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 4.805749416351318
        Y: 4.467410564422607
        Z: -0.6507475972175598
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.6747971177101135
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz_default_plugins)
      Yaw: 4.352385997772217
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1043
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd00000004000000000000015600000375fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d00000375000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f00000375fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d00000375000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000007800000003efc0100000002fb0000000800540069006d0065010000000000000780000002fb00fffffffb0000000800540069006d006501000000000000045000000000000000000000050f0000037500000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Time:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1920
  X: 0
  Y: 0
"""

PATH_DISPLAY_TEMPLATE = """
    - Alpha: 1
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: {color}
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Lines
      Line Width: 0.029999999329447746
      Name: Path_{agent_id}
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 255; 85; 255
      Pose Style: None
      Radius: 0.029999999329447746
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /agent_{agent_id}/path
      Value: true"""

def generate_color(agent_id, max_agents=10):
    """Generate a distinct color for each agent using HSV color space."""
    # Use golden angle to distribute colors evenly around the color wheel
    hue = (agent_id * 137.5) % 360  # Golden angle: 360 * (3 - sqrt(5)) / 2
    saturation = 0.8
    value = 0.9
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    
    # Convert to RGB values in range 0-255
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)
    
    return f"{r_int}; {g_int}; {b_int}"

def get_color_rgb_tuple(agent_id, max_agents=10):
    """Get RGB color tuple for an agent."""
    # Use golden angle to distribute colors evenly around the color wheel
    hue = (agent_id * 137.5) % 360  # Golden angle: 360 * (3 - sqrt(5)) / 2
    saturation = 0.8
    value = 0.9
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    
    # Convert to RGB values in range 0-255
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)
    
    return (r_int, g_int, b_int)

def create_colored_box(r, g, b, text=""):
    """Create a colored box using ANSI escape codes."""
    # ANSI escape code for background color
    return f"\033[48;2;{r};{g};{b}m  \033[0m {text}"

def create_colored_text(r, g, b, text):
    """Create colored text using ANSI escape codes."""
    # ANSI escape code for foreground color
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def discover_active_agents():
    """
    Attempt to discover active agents in the ROS2 network.
    Returns a list of agent IDs or None if discovery fails.
    """
    try:
        import rclpy
        from rclpy.node import Node
        import re
        import threading
        import time
        
        class AgentDiscoveryNode(Node):
            def __init__(self):
                super().__init__('agent_discovery_temp')
                self.discovered_agents = set()
                self.discovery_complete = threading.Event()
                
                # Give some time for discovery
                self.timer = self.create_timer(1.0, self.discover_agents)
                self.discovery_timer = self.create_timer(5.0, self.complete_discovery)
                
            def discover_agents(self):
                """Discover active agent nodes"""
                node_names = self.get_node_names()
                agent_pattern = re.compile(r'agent[_\-]?(\d+)', re.IGNORECASE)
                
                for node_name in node_names:
                    match = agent_pattern.search(node_name)
                    if match:
                        agent_id = int(match.group(1))
                        self.discovered_agents.add(agent_id)
                        
            def complete_discovery(self):
                """Complete the discovery process"""
                self.discovery_complete.set()
        
        # Initialize ROS2
        rclpy.init()
        
        try:
            # Create discovery node
            discovery_node = AgentDiscoveryNode()
            
            # Spin for a short time to allow discovery
            start_time = time.time()
            while not discovery_node.discovery_complete.is_set() and (time.time() - start_time) < 6.0:
                rclpy.spin_once(discovery_node, timeout_sec=0.1)
            
            # Get discovered agents
            discovered_agents = sorted(list(discovery_node.discovered_agents))
            
            # Clean up
            discovery_node.destroy_node()
            rclpy.shutdown()
            
            return discovered_agents if discovered_agents else None
            
        except Exception as e:
            print(f"Error during agent discovery: {e}")
            try:
                rclpy.shutdown()
            except:
                pass
            return None
            
    except ImportError:
        print("ROS2 Python libraries not available. Cannot discover agents.")
        return None
    except Exception as e:
        print(f"Failed to discover agents: {e}")
        return None

def generate_path_displays(agent_ids):
    """Generate path display configurations for the given agent IDs."""
    path_displays = []
    
    for agent_id in agent_ids:
        color = generate_color(agent_id)
        path_display = PATH_DISPLAY_TEMPLATE.format(
            agent_id=agent_id,
            color=color
        )
        path_displays.append(path_display)
    
    return "".join(path_displays)

def generate_rviz_config(agent_ids, output_file):
    """Generate the complete RViz configuration file."""
    # Generate path displays
    path_displays = generate_path_displays(agent_ids)
    
    # Fill in the template
    config_content = RVIZ_CONFIG_TEMPLATE.format(path_displays=path_displays)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    print(f"Generated RViz configuration for agents {agent_ids}")
    print(f"Configuration saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate modular RViz configuration for ergodic exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --max-agents 5
  %(prog)s --discover --output my_config.rviz
  %(prog)s --agents 1 2 3 4 5
        """
    )
    
    # Agent specification options (mutually exclusive)
    agent_group = parser.add_mutually_exclusive_group()
    agent_group.add_argument(
        '--max-agents', 
        type=int, 
        default=10,
        help='Maximum number of agents to configure (generates agents 1 to N)'
    )
    agent_group.add_argument(
        '--agents', 
        nargs='+', 
        type=int,
        help='Specific agent IDs to configure (e.g., --agents 1 3 5 7)'
    )
    agent_group.add_argument(
        '--discover', 
        action='store_true',
        help='Attempt to discover active agents in ROS2 network'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='auto_rviz_config.rviz',
        help='Output RViz configuration file'
    )
    
    # Verbosity options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress informational output'
    )
    
    args = parser.parse_args()
    
    # Determine agent IDs
    if args.discover:
        if not args.quiet:
            print("Attempting to discover active agents in ROS2 network...")
        
        discovered_agents = discover_active_agents()
        
        if discovered_agents:
            agent_ids = discovered_agents
            if not args.quiet:
                print(f"Discovered agents: {agent_ids}")
        else:
            if not args.quiet:
                print("No agents discovered. Falling back to default configuration.")
            agent_ids = list(range(1, args.max_agents + 1))
            
    elif args.agents:
        agent_ids = sorted(args.agents)
        if not args.quiet:
            print(f"Using specified agents: {agent_ids}")
            
    else:
        agent_ids = list(range(1, args.max_agents + 1))
        if not args.quiet:
            print(f"Generating configuration for agents 1 to {args.max_agents}")
    
    # Validate agent IDs
    if not agent_ids:
        print("Error: No agent IDs specified or discovered.")
        return 1
    
    if any(agent_id <= 0 for agent_id in agent_ids):
        print("Error: Agent IDs must be positive integers.")
        return 1
    
    # Generate configuration
    try:
        generate_rviz_config(agent_ids, args.output)
        
        if not args.quiet:
            print(f"\nTo use this configuration with RViz:")
            print(f"  rviz2 -d {args.output}")
            print(f"\nOr load it manually in RViz: File -> Open Config -> {args.output}")
            
            # Print color mappings with visual colors
            print(f"\nAgent color mappings:")
            print("  (Colors as they will appear in RViz)")
            for agent_id in agent_ids:
                color_str = generate_color(agent_id)
                r, g, b = get_color_rgb_tuple(agent_id)
                
                # Create colored box and text
                colored_box = create_colored_box(r, g, b)
                colored_agent_text = create_colored_text(r, g, b, f"Agent {agent_id}")
                
                print(f"  {colored_box} {colored_agent_text}: RGB({color_str})")
        
        return 0
        
    except Exception as e:
        print(f"Error generating configuration: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

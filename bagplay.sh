#!/bin/bash

# ROS Bag Player with Dummy Agent Nodes
# This script spawns dummy nodes named agent_1, agent_2, etc. and plays a ROS bag
# All processes are terminated when Ctrl+C is pressed

# Default values
BAG_PATH=""
NUM_AGENTS=2
AGENT_PREFIX="agent_"
RATE=""
LOOP=false
DELAY=""
START_OFFSET=""
ROSBAG_OPTIONS=""

# Array to store process IDs
declare -a PIDS=()

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS] <bag_path>"
    echo "Options:"
    echo "  -n, --num-agents NUM    Number of dummy agent nodes to spawn (default: 2)"
    echo "  -p, --prefix PREFIX     Prefix for agent node names (default: 'agent_')"
    echo "  -r, --rate RATE         Playback rate multiplier (e.g., 2.0 for 2x speed)"
    echo "  -l, --loop              Loop playback indefinitely"
    echo "  -d, --delay DELAY       Delay in seconds before starting playback"
    echo "  -s, --start-offset SEC  Start playback from offset (seconds)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -n 3 rosbag/test"
    echo "  $0 -r 2.0 --loop rosbag/airplane_hermes_1"
    echo "  $0 --num-agents 4 --prefix drone_ -r 0.5 rosbag/drones_in_small_space"
}

# Function to cleanup processes
cleanup() {
    echo ""
    echo "Terminating all processes..."
    
    # First, send SIGTERM to all processes for graceful shutdown
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Sending SIGTERM to process $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # Give processes time to shut down gracefully
    echo "Waiting for graceful shutdown..."
    sleep 2
    
    # Check which processes are still running and force kill if necessary
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    
    echo "Cleanup completed."
    exit 0
}

# Function to cleanup on script exit (normal or interrupted)
cleanup_on_exit() {
    echo ""
    echo "Cleaning up dummy agent processes..."
    
    # Kill dummy agent processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating dummy agent process $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # Give processes time to shut down gracefully
    sleep 1
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    
    echo "Cleanup completed."
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM
trap cleanup_on_exit EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        -p|--prefix)
            AGENT_PREFIX="$2"
            shift 2
            ;;
        -r|--rate)
            RATE="$2"
            shift 2
            ;;
        -l|--loop)
            LOOP=true
            shift
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        -s|--start-offset)
            START_OFFSET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
        *)
            BAG_PATH="$1"
            shift
            ;;
    esac
done

# Check if bag path is provided
if [[ -z "$BAG_PATH" ]]; then
    echo "Error: Bag path is required"
    usage
    exit 1
fi

# Check if bag path exists
if [[ ! -d "$BAG_PATH" ]]; then
    echo "Error: Bag path '$BAG_PATH' does not exist"
    exit 1
fi

echo "Starting ROS bag playback with dummy agents..."
echo "Bag path: $BAG_PATH"
echo "Number of agents: $NUM_AGENTS"
echo "Agent prefix: $AGENT_PREFIX"
if [[ -n "$RATE" ]]; then
    echo "Playback rate: ${RATE}x"
fi
if [[ "$LOOP" == true ]]; then
    echo "Loop mode: enabled"
fi
if [[ -n "$DELAY" ]]; then
    echo "Start delay: ${DELAY} seconds"
fi
if [[ -n "$START_OFFSET" ]]; then
    echo "Start offset: ${START_OFFSET} seconds"
fi
echo "Press Ctrl+C to stop all processes"
echo ""

# Source ROS setup if available
if [[ -f "install/setup.bash" ]]; then
    source install/setup.bash
    echo "Sourced local ROS workspace"
elif [[ -f "/opt/ros/humble/setup.bash" ]]; then
    source /opt/ros/humble/setup.bash
    echo "Sourced ROS Humble"
elif [[ -f "/opt/ros/foxy/setup.bash" ]]; then
    source /opt/ros/foxy/setup.bash
    echo "Sourced ROS Foxy"
fi

# Start dummy agent nodes
echo "Spawning dummy agent nodes..."
for ((i=1; i<=NUM_AGENTS; i++)); do
    node_name="${AGENT_PREFIX}${i}"
    echo "Starting dummy node: $node_name"
    
    # Start our custom dummy agent node
    python3 src/ergodic_exploration/ergodic_exploration/dummy_agent.py "$node_name" &
    PIDS+=($!)
    
    sleep 0.5  # Small delay to avoid overwhelming the system
done

echo "All dummy agent nodes started."
echo ""

# Build rosbag play command with options
ROSBAG_CMD="ros2 bag play \"$BAG_PATH\""

if [[ -n "$RATE" ]]; then
    ROSBAG_CMD+=" --rate $RATE"
fi

if [[ "$LOOP" == true ]]; then
    ROSBAG_CMD+=" --loop"
fi

if [[ -n "$START_OFFSET" ]]; then
    ROSBAG_CMD+=" --start-offset $START_OFFSET"
fi

# Add delay if specified
if [[ -n "$DELAY" ]]; then
    echo "Waiting $DELAY seconds before starting playback..."
    sleep "$DELAY"
fi

# Start ROS bag playback
echo "Starting ROS bag playback..."
echo "Command: $ROSBAG_CMD"
echo "Keyboard controls available:"
echo "  SPACE: Pause/Resume"
echo "  UP ARROW: Increase rate 10%"
echo "  DOWN ARROW: Decrease rate 10%"
echo "  RIGHT ARROW: Play next message"
echo ""

# Execute rosbag play in foreground so it can access stdin for keyboard controls
eval "$ROSBAG_CMD"
ROSBAG_EXIT_CODE=$?

if [[ $ROSBAG_EXIT_CODE -eq 0 ]]; then
    echo "ROS bag playback completed successfully."
else
    echo "ROS bag playback terminated with exit code: $ROSBAG_EXIT_CODE"
fi

# Normal exit will trigger cleanup_on_exit via EXIT trap
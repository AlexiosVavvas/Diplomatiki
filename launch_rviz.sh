#!/bin/bash
"""
Quick Launcher for RViz Configuration Generation and Launch

This script combines configuration generation and RViz launch into a single command.
It can discover active agents or use a specified number of agents.

Usage:
    ./launch_rviz.sh [OPTIONS]
    
Options:
    --discover          Discover active agents automatically
    --agents N          Configure for N agents (default: 5)
    --specific ID...    Configure for specific agent IDs (e.g., --specific 1 3 5)
    --config FILE       Use existing config file instead of generating
    --no-launch         Only generate config, don't launch RViz
    --help             Show this help message

Examples:
    ./launch_rviz.sh                    # Generate config for 5 agents and launch RViz
    ./launch_rviz.sh --discover         # Discover agents and launch RViz
    ./launch_rviz.sh --agents 3         # Generate config for 3 agents
    ./launch_rviz.sh --specific 1 2 5   # Generate config for agents 1, 2, and 5
    ./launch_rviz.sh --config my.rviz   # Launch RViz with existing config
"""

# Default values
MAX_AGENTS=5
CONFIG_FILE="auto_rviz_config.rviz"
DISCOVER=false
NO_LAUNCH=false
SPECIFIC_AGENTS=""

# Function to show help
show_help() {
    echo "Quick Launcher for RViz Configuration Generation and Launch"
    echo ""
    echo "Usage:"
    echo "    $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "    --discover          Discover active agents automatically"
    echo "    --agents N          Configure for N agents (default: 5)"
    echo "    --specific ID...    Configure for specific agent IDs (e.g., --specific 1 3 5)"
    echo "    --config FILE       Use existing config file instead of generating"
    echo "    --no-launch         Only generate config, don't launch RViz"
    echo "    --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "    $0                    # Generate config for 5 agents and launch RViz"
    echo "    $0 --discover         # Discover agents and launch RViz" 
    echo "    $0 --agents 3         # Generate config for 3 agents"
    echo "    $0 --specific 1 2 5   # Generate config for agents 1, 2, and 5"
    echo "    $0 --config my.rviz   # Launch RViz with existing config"
}

# Parse command line arguments
GENERATE_CONFIG=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --discover)
            DISCOVER=true
            shift
            ;;
        --agents)
            MAX_AGENTS="$2"
            shift 2
            ;;
        --specific)
            shift
            SPECIFIC_AGENTS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SPECIFIC_AGENTS="$SPECIFIC_AGENTS $1"
                shift
            done
            ;;
        --config)
            CONFIG_FILE="$2"
            GENERATE_CONFIG=false
            shift 2
            ;;
        --no-launch)
            NO_LAUNCH=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Check if we need to generate a config
if $GENERATE_CONFIG; then
    echo "🔧 Generating RViz configuration..."
    
    # Build the generation command
    GEN_CMD="python3 generate_rviz_config.py --output $CONFIG_FILE"
    
    if $DISCOVER; then
        GEN_CMD="$GEN_CMD --discover"
        echo "   🔍 Discovering active agents..."
    elif [[ -n "$SPECIFIC_AGENTS" ]]; then
        GEN_CMD="$GEN_CMD --agents$SPECIFIC_AGENTS"
        echo "   🎯 Configuring for agents:$SPECIFIC_AGENTS"
    else
        GEN_CMD="$GEN_CMD --max-agents $MAX_AGENTS"
        echo "   📊 Configuring for $MAX_AGENTS agents"
    fi
    
    # Execute the generation command
    eval $GEN_CMD
    
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to generate RViz configuration"
        exit 1
    fi
    
    echo "✅ Configuration generated: $CONFIG_FILE"
else
    echo "📄 Using existing configuration: $CONFIG_FILE"
    
    # Check if the config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "❌ Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
fi

# Launch RViz if requested
if ! $NO_LAUNCH; then
    echo "🚀 Launching RViz with configuration..."
    echo "   📂 Config file: $CONFIG_FILE"
    
    # Check if rviz2 is available
    if ! command -v rviz2 &> /dev/null; then
        echo "❌ rviz2 command not found. Make sure ROS2 is installed and sourced."
        exit 1
    fi
    
    # Launch RViz
    rviz2 -d "$CONFIG_FILE"
else
    echo "✅ Configuration ready. To launch RViz manually, run:"
    echo "   rviz2 -d $CONFIG_FILE"
fi

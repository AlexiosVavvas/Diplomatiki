#!/bin/bash

# FlightGear Visualizer Launch Script
# This script starts FlightGear with the correct parameters

# Default values
AIRCRAFT="c172p"
AIRPORT="KSFO"
FDM_IN_PORT="5502"
FDM_OUT_PORT="5501"
UPDATE_RATE="30"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --aircraft)
            AIRCRAFT="$2"
            shift 2
            ;;
        --airport)
            AIRPORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --aircraft AIRCRAFT    Aircraft model (default: c172p)"
            echo "  --airport AIRPORT      Starting airport (default: KSFO)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --aircraft ufo --airport KJFK"
            echo ""
            echo "Common aircraft: c172p, 737-300, A320, ufo"
            echo "Common airports: KSFO, KJFK, EGLL, LFPG"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if FlightGear is installed
if ! command -v fgfs &> /dev/null; then
    echo "Error: FlightGear (fgfs) not found in PATH"
    echo "Please install FlightGear first:"
    echo "  sudo apt-get install flightgear"
    exit 1
fi

# Display configuration
echo "========================================="
echo "Starting FlightGear Visualizer"
echo "========================================="
# echo "Aircraft:    $AIRCRAFT"
# echo "Airport:     $AIRPORT"
echo "FDM In:      localhost:$FDM_IN_PORT"
echo "FDM Out:     localhost:$FDM_OUT_PORT"
echo "Update Rate: ${UPDATE_RATE} Hz"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop FlightGear"
echo ""

# Start FlightGear
# IMPORTANT: --fdm=null tells FlightGear to use ONLY external FDM (no internal physics)
fgfs \
    --fdm=null \
    --native-fdm=socket,in,${UPDATE_RATE},localhost,${FDM_IN_PORT},udp \
    --native-fdm=socket,out,${UPDATE_RATE},localhost,${FDM_OUT_PORT},udp \
    --timeofday=noon \
    # --airport=${AIRPORT} \
    # --aircraft=${AIRCRAFT} \
    # --disable-real-weather-fetch \
    # --disable-clouds \
    # --disable-clouds3d \
    # --fog-disable \
    # --disable-random-objects \
    # --disable-ai-models \
    # --disable-ai-traffic
    # --disable-horizon-effect \

# Note: Additional performance options above
# Remove them if you want better visuals but slower performance

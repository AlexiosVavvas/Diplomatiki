#!/bin/bash

# Check if a case name was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <case_name>"
    echo "Example: $0 B1"
    exit 1
fi

CASE_NAME=$1
LAUNCH_DIR="src/ergodic_exploration/launch"

# Check if launch directory exists
if [ ! -d "$LAUNCH_DIR" ]; then
    echo "Error: Directory $LAUNCH_DIR does not exist"
    exit 1
fi

# Copy the template files
echo "Creating case: $CASE_NAME"

cp "$LAUNCH_DIR/fixed_wing_free_fly_agent.yaml" "$LAUNCH_DIR/${CASE_NAME}_agent.yaml"
echo "Created ${CASE_NAME}_agent.yaml"

cp "$LAUNCH_DIR/fixed_wing_free_fly.launch.py" "$LAUNCH_DIR/${CASE_NAME}.launch.py"
echo "Created ${CASE_NAME}.launch.py"

cp "$LAUNCH_DIR/fixed_wing_free_fly_obs.yaml" "$LAUNCH_DIR/${CASE_NAME}_obs.yaml"
echo "Created ${CASE_NAME}_obs.yaml"

# Create an empty changelog file
touch "$LAUNCH_DIR/${CASE_NAME}_changeLog.txt"
echo "Created ${CASE_NAME}_changeLog.txt"

echo ""
echo "Case $CASE_NAME created successfully in $LAUNCH_DIR/"

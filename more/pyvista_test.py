import numpy as np
import pyvista as pv
import time

# --- Example data (replace with your real data) ---
obstacles = [
    ( 1,  1, 0.5, 0.5, 2.0),
    (-1, -1, 0.5, 0.5, 1.5),
    ( 2, -2, 0.5, 0.5, 1.8),
]
targets = [
    ( 0,  2, 0.0),
    (-2,  0, 0.0),
    ( 2,  2, 0.0),
]
t = np.linspace(0, 4*np.pi, 50)
agent_states = []
for phase in [0, np.pi/2]:
    x = 2*np.cos(t+phase)
    y = 2*np.sin(t+phase)
    z = 0.5 + 0.5*np.sin(2*t+phase)
    # Add orientation data: roll, pitch, yaw
    roll = 0.1*np.sin(3*t+phase)  # Small roll oscillation
    pitch = 0.1*np.cos(3*t+phase)  # Small pitch oscillation
    yaw = t+phase  # Continuous yaw rotation
    states = np.vstack([x,y,z,roll,pitch,yaw] + [np.zeros_like(x) for _ in range(6)]).T
    agent_states.append(states)
num_agents = len(agent_states)
num_steps  = agent_states[0].shape[0]

def create_drone_mesh(center=(0,0,0), roll=0, pitch=0, yaw=0, scale=0.4):
    """Create a drone-shaped mesh with orientation"""
    # Main body (cylinder)
    body = pv.Cylinder(center=center, direction=(0,0,1), radius=scale*0.3, height=scale*0.2)
    
    # Propeller arms (4 cylinders in cross pattern)
    arm_length = scale * 0.8
    arm_radius = scale * 0.05
    
    # Create arms along X and Y axes
    arm1 = pv.Cylinder(center=center, direction=(1,0,0), radius=arm_radius, height=arm_length)
    arm2 = pv.Cylinder(center=center, direction=(0,1,0), radius=arm_radius, height=arm_length)
    
    # Propellers (small cylinders at arm ends)
    prop_radius = scale * 0.15
    prop_height = scale * 0.02
    positions = [
        (arm_length/2, 0, 0), (-arm_length/2, 0, 0),
        (0, arm_length/2, 0), (0, -arm_length/2, 0)
    ]
    
    # Combine all parts
    drone = body + arm1 + arm2
    for pos in positions:
        prop_center = (center[0] + pos[0], center[1] + pos[1], center[2] + pos[2])
        prop = pv.Cylinder(center=prop_center, direction=(0,0,1), 
                          radius=prop_radius, height=prop_height)
        drone = drone + prop
    
    # Apply rotations (roll, pitch, yaw)
    if roll != 0:
        drone = drone.rotate_x(np.degrees(roll), point=center)
    if pitch != 0:
        drone = drone.rotate_y(np.degrees(pitch), point=center)
    if yaw != 0:
        drone = drone.rotate_z(np.degrees(yaw), point=center)
    
    return drone

# --- PyVista scene setup in interactive mode ---
plotter = pv.Plotter(window_size=(800,600))
plotter.set_background("white")

# Add ground plane with limits
ground_size = 6  # Adjust size as needed
ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), 
                  i_size=ground_size, j_size=ground_size, 
                  i_resolution=10, j_resolution=10)
plotter.add_mesh(ground, color="lightgray", opacity=0.3, show_edges=True)

# Add ground boundary rectangle
boundary_points = np.array([
    [-ground_size/2, -ground_size/2, 0],
    [ground_size/2, -ground_size/2, 0],
    [ground_size/2, ground_size/2, 0],
    [-ground_size/2, ground_size/2, 0],
    [-ground_size/2, -ground_size/2, 0]  # Close the rectangle
])
boundary_line = pv.Spline(boundary_points, len(boundary_points))
plotter.add_mesh(boundary_line, color="black", line_width=4)

# Add obstacles
for (x, y, w, d, h) in obstacles:
    cube = pv.Cube(center=(x+w/2, y+d/2, h/2),
                   x_length=w, y_length=d, z_length=h)
    plotter.add_mesh(cube, color="black", opacity=1.0)

# Add targets
for (tx, ty, tz) in targets:
    sph = pv.Sphere(radius=0.1, center=(tx,ty,tz))
    plotter.add_mesh(sph, color="red")

# Prepare dynamic actors
colors = ['blue', 'green', 'orange', 'purple', 'cyan'][:num_agents]
lines = []
drones = []
trail_length = 50  # Number of steps to show in trail

# Initialize with first timestep
for i in range(num_agents):
    # Initial trail (just the starting point)
    pts = agent_states[i][:1, :3]
    line = pv.Line(pts[0], pts[0])
    lines.append(plotter.add_mesh(line, color=colors[i], line_width=3))
    
    # Create drone with initial orientation
    initial_state = agent_states[i][0]
    drone_mesh = create_drone_mesh(
        center=initial_state[:3],
        roll=initial_state[3],
        pitch=initial_state[4],
        yaw=initial_state[5]
    )
    drones.append(plotter.add_mesh(drone_mesh, color=colors[i]))

# Show the plot and start animation
plotter.show(auto_close=False, interactive_update=True)

# Animation loop
for step in range(num_steps):
    for i in range(num_agents):
        # Update trail - show path from current position backward
        start_idx = max(0, step - trail_length)
        end_idx = step + 1
        pts = agent_states[i][start_idx:end_idx, :3]
        
        if len(pts) > 1:
            new_line = pv.Spline(pts, len(pts))
        else:
            new_line = pv.Line(pts[0], pts[0])
        
        plotter.remove_actor(lines[i])
        lines[i] = plotter.add_mesh(new_line, color=colors[i], line_width=3)
        
        # Update drone position and orientation
        plotter.remove_actor(drones[i])
        current_state = agent_states[i][step]
        new_drone = create_drone_mesh(
            center=current_state[:3],
            roll=current_state[3],
            pitch=current_state[4],
            yaw=current_state[5]
        )
        drones[i] = plotter.add_mesh(new_drone, color=colors[i])
    
    # Update the plot
    plotter.update()
    
    # Add small delay for animation speed
    time.sleep(0.05)

plotter.close()

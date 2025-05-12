# cd one dir back
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------------------------------------
import numpy as np
from my_erg_lib.model_dynamics import Quadcopter

# x0 = [x,   y,   z, ψ, θ, φ, x', y', z', ψ', θ', φ']
x0 =   [0.5, 0.4, 2, 30*3.14/180, 30*3.14/180, 30*3.14/180, 0,  0,  -1,  0,  0.5,  0]

q = Quadcopter(dt=0.005, mass=0.1, damping=0, x0=x0)

states_list = []
time_list = []
input_list = []
conv_inp_list = []

t_now = 0
for i in range(2000):
    
    # print A and B matrices
    u = q.calcLQRcontrol(q.state, t=t_now, z_target=4)
    q.state = q.step(q.state, u)
    print(q.state[:3])

    states_list.append(q.state.copy())
    time_list.append(t_now + q.dt)
    input_list.append(u.copy())
    conv_inp_list.append(q.convertInputToMotorCommands(u))
    t_now += q.dt

    if q.state[2] < 0 or q.state[2] > 100:
        print("Quadcopter crashed!")
        break


# keep all the states up to the last time
states_list = np.array(states_list[:])
time_list = np.array(time_list[:])
input_list = np.array(input_list[:])
conv_inp_list = np.array(conv_inp_list[:])


# Visualise the quad results ------------------------------------------------------------------------
from vis import plotQuadTrajWithInputs
plotQuadTrajWithInputs(time_list, states_list, input_list, conv_inp_list=conv_inp_list, quad_model=q)
from vis import animateQuadcopter
animateQuadcopter(time_list, states_list)
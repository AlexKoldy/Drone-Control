import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from rotations import *
import mavsim_python_parameters_aerosonde_parameters as P
import simulation_parameters as SIM
from autopilot import autopilot
from msg_state import msgState
from signals import *

plt.close('all')

# Sim time
sim_time = 100

from msg_autopilot import msgAutopilot
commands = msgAutopilot()
Va_command = signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=0.0,
                     frequency=0.01)
h_command = signals(dc_offset=100.0,
                    amplitude=10.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = signals(dc_offset=np.radians(180),
                      amplitude=np.radians(45),
                      start_time=5.0,
                      frequency=0.015)







# Trim state from Beard (Ch05)
#  pn, pe, pd, u, v, w,  e0, e1, e2, e3, p, q, r
x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.971443, 0.000000, 1.194576, 0.993827, 0.000000, 0.110938, 0.000000, 0.000000, 0.000000, 0.000000]]).flatten()
#  delta_e, delta_a, delta_r, delta_t
delta_trim = np.array([[-0.118662, 0.009775, -0.001611, 0.857721]]).flatten()


# Init drone
drone = Drone()
ctrl = autopilot(SIM.ts_simulation)

# Set drone rigid body state to x_trim
drone.state.rigid_body = x_trim
# Set drone input to delta_trim 
delta = delta_trim

# Drone Sim
t_history = [0]
x_history = [drone.state.rigid_body]
Va_history = []
delta_history = []
chi_history = []

while drone.state.time <= sim_time:
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)
    delta, commanded_state = ctrl.update(cmd=commands, state=drone.state)
    drone.update(delta)
    t_history.append(drone.state.time)
    x_history.append(drone.state.rigid_body)
    Va_history.append(np.linalg.norm(drone.state.rigid_body[3:6]))
    delta_history.append(delta)
    R = quat2rot(drone.state.rigid_body[6:10])
    Vg_inertial = R @ drone.state.rigid_body[3:6]
    chi = np.arctan2(Vg_inertial[1], Vg_inertial[0])
    chi_history.append(chi)
    
# Convert to numpy array
x_history = np.asarray(x_history)
plt.figure()
plt.plot(t_history, x_history[:, :3])
plt.legend(['x1', 'x2', 'x3'])
plt.show()

plt.figure()
plt.plot(t_history, x_history[:, 3:6])
plt.legend(['u', 'v', 'w'])
plt.show()




# Convert quat to euler
nsteps = x_history.shape[0]
euler_angles = np.zeros((nsteps, 3))
for i in range(nsteps):         
    e = x_history[i, 6:10]
    euler_angles[i, :] = quat2euler(e) # phi, theta, psi = 

end_time = len(x_history)-1

plt.figure()
plt.plot(t_history, (180/np.pi)*euler_angles)
plt.legend(['phi', 'theta', 'psi'])
plt.show()

plt.figure()
plt.plot(t_history, x_history[:, 10:])
plt.legend(['p', 'q', 'r'])
plt.show()

plt.figure()
plt.plot(t_history[:end_time], Va_history[:end_time], [0, t_history[end_time]], [28, 28])
plt.legend(['Airspeed', 'Setpoint'])
plt.show()

plt.figure()
plt.plot(t_history[:end_time], -1 * x_history[:end_time, 2], [0, t_history[end_time]], [commands.altitude_command, commands.altitude_command])
plt.legend(['Altitude', 'Setpoint'])
plt.show

plt.figure()
plt.plot(t_history, (180/np.pi)*euler_angles[:,1])
plt.legend(['theta'])
plt.show()

plt.figure()
plt.plot(t_history[:end_time], chi_history[:end_time], [0, t_history[end_time]], [commands.course_command - (2*np.pi), commands.course_command - (2*np.pi)])
plt.legend(['chi', 'Setpoint'])
plt.show()  

plt.figure()
plt.plot(t_history[:end_time], np.asarray(delta_history)[:end_time])
plt.legend(['elevator', 'aileron', 'rudder', 'thrust'])
plt.show()

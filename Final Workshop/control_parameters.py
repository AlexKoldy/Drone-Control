import sys
import numpy as np
from model_coef import *
from mavsim_python_parameters_aerosonde_parameters import *

sigma = 1  # low pass filter gain for derivative
Va0 = Va_trim
Va = Va_trim
Vg = Va

#roll
a_roll_1 = -0.5 * rho * Va**2 * S_wing * b * C_p_p * b / (2 * Va)
a_roll_2 = 0.5 * rho * Va**2 * S_wing * b * C_p_delta_a
#pitch
a_pitch_constant = rho * (Va**2) * c * S_wing / (2 * Jy)
a_pitch_1 = -1 * (a_pitch_constant * C_m_q * (c / (2*Va) )) #
a_pitch_2 = -1 * (a_pitch_constant * C_m_alpha) 
a_pitch_3 = a_pitch_constant * C_m_delta_e
#airspeed
a_airspeed_1=(rho*Va_trim*S_wing/mass*(C_D_0+C_D_alpha*alpha_trim+C_D_delta_e*u_trim[0]))+(rho*S_prop/mass*Va_trim*c_prop)
a_airspeed_2=rho*S_prop/mass*c_prop*(k_motor**2)*u_trim[3]
a_airspeed_3=gravity*np.cos(theta_trim-alpha_trim)

#----------roll loop-------------
# get transfer function data for delta_a to phi
delta_a_max = np.radians(45)
e_roll_max = np.radians(60)
wn_roll = np.sqrt( a_roll_2 * delta_a_max / e_roll_max)
zeta_roll = 0.707
roll_kp = delta_a_max / e_roll_max
roll_kd = ( (2 * zeta_roll * wn_roll) - a_roll_1 ) / a_roll_2

#----------course loop-------------
wn_course = wn_roll / 10
zeta_course = 0.707
course_kp = 2 * zeta_course * wn_course * Vg / gravity
course_ki = ( wn_course**2 * Vg ) / gravity

#----------yaw damper-------------
yaw_damper_tau_r = 1/0.45
yaw_damper_kp = 0.196

#----------pitch loop-------------
delta_e_max = np.radians(30)
e_pitch_max = np.radians(60)
wn_pitch = 1.5 #(a_pitch_2 + (np.abs(a_pitch_3)) * (delta_e_max / e_pitch_max))**0.5
zeta_pitch = 0.707
pitch_kp = (delta_e_max / e_pitch_max) * np.sign(a_pitch_3)#( ( wn_pitch**2 ) - a_pitch_2 ) / a_pitch_3
pitch_kd = ( ( 2 * zeta_pitch * wn_pitch ) - a_pitch_1 ) / a_pitch_3
K_theta_DC = ( pitch_kp * a_pitch_3 ) / ( a_pitch_2 + ( pitch_kp * a_pitch_3 ) )

#----------altitude loop-------------
Wh = 12
wn_altitude = wn_pitch / Wh
zeta_altitude = 0.707
altitude_kp = ( 2 * zeta_altitude * wn_altitude ) / ( K_theta_DC * Va ) 
altitude_ki = ( wn_altitude**2 ) / ( K_theta_DC * Va )
altitude_zone = 1000  # moving saturation limit around current altitude

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 2.5
zeta_airspeed_throttle = 0.707
airspeed_throttle_kp = ( ( 2 * zeta_airspeed_throttle * wn_airspeed_throttle ) - a_airspeed_1 ) / a_airspeed_2
airspeed_throttle_ki = ( wn_airspeed_throttle**2 ) / ( a_airspeed_2 )

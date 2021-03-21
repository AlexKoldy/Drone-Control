import sys
import numpy as np
import control_parameters as AP
from transfer_function import transferFunction
from wrap import wrap
from pid_control import pidControl, piControl, pdControlWithRate
from msg_state import msgState

class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pdControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = piControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = transferFunction(
                        num=np.array([[AP.yaw_damper_kp, 0]]),
                        den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
                        Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = pdControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = piControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = piControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=10.0)
        self.commanded_state = msgState()

    def update(self, cmd, state):

        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        print('State Chi: ', state.chi, 'Chi_C: ', chi_c)
        phi_c = self.saturate(cmd.phi_feedforward + self.course_from_roll.update(chi_c, state.chi), - np.radians(30), np.radians(30))
        print('Phi C: ', phi_c)
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.rigid_body[10])
        delta_r = self.yaw_damper.update(state.rigid_body[12])

        # longitudinal autopilot
        # saturate the altitude command
        h_c = self.saturate(cmd.altitude_command, state.rigid_body[2] - AP.altitude_zone, state.rigid_body[2] + AP.altitude_zone)
        print('h_c: ', h_c, 'h: ', -state.rigid_body[2], 'altitude command: ', cmd.altitude_command)
        theta_c = self.altitude_from_pitch.update(h_c, -state.rigid_body[2])
        print(theta_c)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.rigid_body[11])
        # print('delta e', delta_e)
        delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.airspeed)

        # construct output and commanded states
        delta = np.array([[delta_e], [delta_a], [delta_r], [delta_t]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta.flatten(), self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output

import numpy as np
import mavsim_python_parameters_aerosonde_parameters as P

class Thrust:
    def __init__(self, state):
        self.state = state
        # self.atmosphere = ISA()?
        self.delta_t = 0
        self.force_moment = np.zeros(6)

    def update(self, delta_t):
        self.delta_t = delta_t
        # Compute force and moment
        self.compute_force_moment()

    def compute_force_moment(self):
        # compute thrust and torque due to propeller (See addendum by McLain)
        # map delta t throttle command(0 to 1) into motor input voltage
        V_in = P.V_max * self.delta_t
        Va = self.state.airspeed
        # Quadratic formula to solve for motor speed
        a = P.rho*np.power(P.D_prop, 5)*P.C_Q0/(2*np.pi)**2
        b = P.rho*np.power(P.D_prop, 4)*P.C_Q1*Va/(2*np.pi) \
            + P.KQ**2/P.R_motor
        c = P.rho*np.power(P.D_prop, 3)*P.C_Q2*Va**2 \
            - P.KQ*V_in/P.R_motor + P.KQ*P.i0
        # Consider only positive root
        Omega_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        # compute advance r a t i o
        J = 2*np.pi*Va/(Omega_p*P.D_prop)
        # compute nondimensionalized coefficients of thrust and torque
        C_T = P.C_T2*J**2 + P.C_T1*J + P.C_T0
        C_Q = P.C_Q2*J**2 + P.C_Q1*J + P.C_Q0
        # add thrust and torque due to propeller
        T_p = P.rho*np.power(P.D_prop, 4)*Omega_p**2*C_T/(4*np.pi**2)
        Q_p = P.rho*np.power(P.D_prop, 5)*Omega_p**2*C_Q/(4*np.pi**2)
        self.force_moment[0] = T_p
        self.force_moment[3] = Q_p

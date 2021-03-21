import numpy as np
import mavsim_python_parameters_aerosonde_parameters as P
from rotations import rot2, rot3

class Aerodynamics:
    def __init__(self, state):
        self.state = state
        # self.atmosphere = ISA()?
        self.delta_e = 0
        self.delta_a = 0
        self.delta_r = 0
        self.force_moment = np.zeros(6)

    def update(self, delta):
        # Split up inputs delta (elevator, aileron, rudder)
        self.delta_e, self.delta_a, self.delta_r = delta
        # Get air velocity and speed
        ur, vr, wr = self.state.air_velocity
        Va = self.state.airspeed
        # Compute angle of attack and sideslip angle        
        self.state.alpha = np.arctan2(wr, ur)
        self.state.beta = np.arcsin(vr/Va)
        # Compute force and moment
        self.compute_force()
        self.compute_moment()

    @property
    def sigma(self):
        a = np.exp(-P.M*(self.state.alpha - P.alpha0))
        b = np.exp( P.M*(self.state.alpha + P.alpha0))
        return (1 + a + b)/((1 + a)*(1 + b))        
 
    @property
    def C_L_flatplate(self):
        return 2*np.sign(self.state.angle_of_attack)*np.sin(self.state.angle_of_attack)**2 \
            *np.cos(self.state.angle_of_attack)
    
    def compute_force(self):
        Va = self.state.airspeed
        alpha = self.state.angle_of_attack
        beta = self.state.sideslip_angle
        # Nondimensionalized rates p, q, r
        p = self.state.p*P.b/(2*Va) # nondim. p
        q = self.state.q*P.c/(2*Va) # nondim. q
        r = self.state.r*P.b/(2*Va) # nondim. r
        # Dynamic pressure
        p_dyn = 0.5*P.rho*Va**2
        # Lift Coefficient, CL
        C_L_lin = P.C_L_0 + P.C_L_alpha*alpha
        C_L = (1 - self.sigma)*C_L_lin + self.sigma*self.C_L_flatplate \
            + P.C_L_q*q + P.C_L_delta_e*self.delta_e
        # Drag Coefficient, CD   
        C_D = P.C_D_p \
            + (P.C_L_0 + P.C_L_alpha*alpha)**2/(np.pi*P.e*P.AR) \
            + P.C_D_q*q +P.C_D_delta_e*self.delta_e

        # # linear CD(alpha)
        # C_D = P.C_D_0 + P.C_D_alpha*self.state.alpha \
        #     + P.C_D_q*q +P.C_D_delta_e*self.delta_e

        # Sideforce Coefficient, CY
        C_Y = P.C_Y_0 + P.C_Y_beta*beta \
            + P.C_Y_p*p + P.C_Y_r*r \
            + P.C_Y_delta_a*self.delta_a + P.C_Y_delta_r*self.delta_r
        # Lift, Drag, Side force
        F_L = C_L * p_dyn * P.S_wing
        F_D = C_D * p_dyn * P.S_wing
        F_Y = C_Y * p_dyn * P.S_wing

        # Convert Lift and Drag to body frame: 1, 3 direction
        #
        # ERROR IN BOOK, rotation BETA missing:
        # All aero forces are defined in wind frame NOT stability 
        # R_bw = rot2(-self.state.alpha) @ rot3(self.state.beta)
        R = rot2(-alpha) 
        F = R @ np.array([-F_D, F_Y, -F_L])
        self.force_moment[ :3] = F.flatten()
        
    def compute_moment(self):
        Va = self.state.airspeed
        alpha = self.state.angle_of_attack
        beta = self.state.sideslip_angle        
        # Nondimensionalized rates p, q, r
        p = self.state.p*P.b/(2*Va) # nondim. p
        q = self.state.q*P.c/(2*Va) # nondim. q
        r = self.state.r*P.b/(2*Va) # nondim. r
        # Dynamic pressure
        p_dyn = 0.5*P.rho*Va**2
        # Rolling Moment, C_l       
        C_ell = P.C_ell_0 + P.C_ell_beta*beta \
              + P.C_ell_p*p + P.C_ell_r*r \
              + P.C_ell_delta_a*self.delta_a \
              + P.C_ell_delta_r*self.delta_r
        # Pitching Moment, C_m
        C_m = P.C_m_0 + P.C_m_alpha*alpha \
            + P.C_m_q*q + P.C_m_delta_e*self.delta_e
        # Yawing Moment, C_n
        C_n = P.C_n_0 + P.C_n_beta*beta \
            + P.C_n_p*p + P.C_n_r*r \
            + P.C_n_delta_a*self.delta_a \
            + P.C_n_delta_r*self.delta_r

        # Roll, pitch, yaw moment
        ell = C_ell * p_dyn * P.S_wing * P.b
        m   = C_m   * p_dyn * P.S_wing * P.c
        n   = C_n   * p_dyn * P.S_wing * P.b
        self.force_moment[3:6] = (ell, m, n)
            
    
import numpy as np
from rotations import rot2, rot3
import mavsim_python_parameters_aerosonde_parameters as P

class Gravity:
    def __init__(self, state):
        self.mass = P.mass
        self.gravity = P.gravity
        self.state = state
        
        
    # Aero quantities
    @property
    def force(self):
        R_ib = self.state.rot
        R_bi = R_ib.T
        W_i = np.array([0, 0, P.mass*P.gravity])
        F = R_bi @ W_i
        return F.flatten()
    
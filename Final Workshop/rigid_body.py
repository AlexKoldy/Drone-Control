import numpy as np
import mavsim_python_parameters_aerosonde_parameters as P
from rotations import quat2rot, prod, skew

class RigidBody:
    """Setup state-space system for integration of rigid body
    """
    def __init__(self):
        self.m = P.mass # mass, kg
        # mass moment of inertia wrt COM in body-fixed frame, m^2 kg
        self.J_bb = np.array([
                    [  P.Jx,    0, -P.Jxz],
                    [     0, P.Jy,      0],
                    [-P.Jxz,    0,   P.Jz]
                    ])
        self.Jinv_bb = np.linalg.inv(self.J_bb)
    
    def __call__(self, t, x, forces_moments):
        return self.eom(t, x, forces_moments)
    
    def eom(self, t, x, forces_moments):
        """EOM: Evaluates f in xdot = f(t, x, forces_moments)
        """
        # Get forces, moments
        F_b = forces_moments[:3] 
        M_b = forces_moments[3:] 
        # Get positions, Euler angles, velocities, angular velocities 
        #r_i  = x[  : 3] # position in i-frame
        v_b  = x[ 3: 6] # velocity in b-frame
        q_ib = x[ 6:10] # quaternion to keep track of angle / axis
        w_b  = x[10:  ] # angular velocity in b-frame
    
        # Normalize quat. -> rotation
        q_ib = q_ib/np.linalg.norm(q_ib) # normalize
        R_ib = quat2rot(q_ib)    

        # Compute equations of motion
        # d/dt(r_i) 
        rdot_i = R_ib @ v_b
        
        # d/dt(q_ib)
        wq_ib = np.zeros(4)
        wq_ib[1:] = w_b.flatten()
        qdot_ib = 0.5 * prod(q_ib, wq_ib)
    
        # Compute derivatives from EOM
        # d/dt(v_b)
        w_bb = skew(w_b)
        vdot_b = F_b/self.m - w_bb @ v_b
        # d/dt(w_b)
        wdot_b = self.Jinv_bb @ (M_b - w_bb @ self.J_bb @ w_b)
        return np.concatenate([
            rdot_i, # d/dt position in i-frame
            vdot_b, # d/dt velocity in b-frame 
            qdot_ib, # d/dt quaternion
            wdot_b  # d/dt angular velocity in b-frame 
            ])
    

import numpy as np
from rotations import quat2rot, quat2euler

class State():
    def __init__(self):
        self.time = 0.
        self.timestep = 0.
        # position in inertial NED frame
        self.pn = 0.      # inertial north position in meters
        self.pe = 0.      # inertial east position in meters
        self.h = 0.      # inertial down position in meters
        # velocity in body frame
        self.u = 0.      # u ground velocity in m/s
        self.v = 0.      # v ground velocity in m/s
        self.w = 0.      # w ground velocity in m/s
        # quaternion e for orientation
        self.e0 = 0.      
        self.e1 = 0.      
        self.e2 = 0.      
        self.e3 = 0.      
        # angular rates in body frame
        self.p = 0.       # roll rate in radians/sec
        self.q = 0.       # pitch rate in radians/sec
        self.r = 0.       # yaw rate in radians/sec
        # wind quantities
        self.wn = 0.      # inertial windspeed in north direction in meters/sec
        self.we = 0.      # inertial windspeed in east direction in meters/sec
        self.uw = 0.      # u wind-velocity in m/s
        self.vw = 0.      # v wind-velocity in m/s
        self.ww = 0.      # w wind-velocity in m/s
        # aero quantities
        self.ur = 0       # u air-velocity in m/s
        self.vr = 0       # v air-velocity in m/s
        self.wr = 0       # w air-velocity in m/s
        self.alpha = 0.   # angle of attack in radians
        self.beta = 0.    # sideslip angle in radians
        # course, path angles        
        self.gamma = 0.   # flight path angle in radians
        #self.chi = 0.     # course angle in radians
        
    # Rigid body state: get, set        
    @property
    def rigid_body(self):
        return np.array([self.pn, self.pe, self.pd,
                self.u, self.v, self.w,
                self.e0, self.e1, self.e2, self.e3,
                self.p, self.q, self.r])
        
    @rigid_body.setter
    def rigid_body(self, x):
        self.pn, self.pe, self.pd = x[ :3]
        self.u, self.v, self.w = x[3:6]
        self.e0, self.e1, self.e2, self.e3 = x[6:10]
        self.p, self.q, self.r = x[10: ]

    @property
    def ground_velocity(self):
        return np.array([self.u, self.v, self.w])
       
    @ground_velocity.setter
    def ground_velocity(self, x):
        self.u, self.v, self.w = x

    @property
    def angular_velocity(self):
        return np.array([self.p, self.q, self.r])

    @angular_velocity.setter
    def angular_velocity(self, x):
        self.p, self.q, self.r = x

   # Quaternion
    @property
    def quaternion(self):
        return np.array([self.e0, self.e1, self.e2, self.e3])
        
    @quaternion.setter
    def quaternion(self, e):
        self.e0, self.e1, self.e2, self.e3 = e
    # Euler angles
    @property
    def euler_angles(self):
        e = self.quaternion
        return quat2euler(e)
    
    @property
    def rot(self):
        return quat2rot(self.quaternion) 

    @property
    def phi(self):
        return self.euler_angles[0]
        
    @property
    def theta(self):
        return self.euler_angles[1]

    @property
    def psi(self):
        return self.euler_angles[2]
    
    @property
    def chi(self):
         R = quat2rot(self.rigid_body[6:10])
         Vg_inertial = R @ self.rigid_body[3:6]
         return np.arctan2(Vg_inertial[1], Vg_inertial[0])

    # Wind state: get, set        
    @property
    def wind_velocity(self):
        return np.array([self.uw, self.vw, self.ww])
    
    @wind_velocity.setter
    def wind_velocity(self, x):
        self.uw, self.vw, self.ww = x

    # Aero quantities
    @property
    def air_velocity(self):
        self.ur = self.u - self.uw
        self.vr = self.v - self.vw
        self.wr = self.w - self.ww
        return np.array([self.ur, self.vr, self.wr])
        
    @property
    def airspeed(self):
        return np.linalg.norm(self.air_velocity)        
        
    @property
    def angle_of_attack(self):
        return self.alpha

    @angle_of_attack.setter
    def angle_of_attack(self, alpha):
        self.alpha = alpha
    
    @property
    def sideslip_angle(self):
        return self.beta

    @sideslip_angle.setter
    def sideslip_angle(self, beta):
        self.beta = beta
        

    
    
    

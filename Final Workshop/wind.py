"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
#import sys
#sys.path.append('..')
#from tools.transfer_function import transferFunction
import numpy as np
from rotations import quat2rot

class Wind:
    def __init__(self, state):
        self.state = state
        
    def update(self):
        """
        Return wind in m/s in body frame
        uw, vw, ww
        """       
        # # steady_state in NED frame
        # # gust in body frame
        # R_bi = quat2rot(state.quaternion).T
        # steady_wind_i = np.zeros(3)
        # gust_b = np.zeros(3) # implement Dryden gusts
        # # Update wind in body frame
        # self.state.wind = R_bi*steady_wind_i + gust_b
        self.state.wind_velocity = np.zeros(3)

        
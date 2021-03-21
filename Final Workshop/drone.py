from state import State
from rigid_body import RigidBody
from gravity import Gravity
from aerodynamics import Aerodynamics
from thrust import Thrust
from wind import Wind
from integrators import get_integrator
from simulation_parameters import ts_simulation

class Drone():
    def __init__(self):
        self.state = State()
        self.state.timestep = ts_simulation
        self.rigid_body = RigidBody()
        self.gravity = Gravity(self.state)
        self.wind = Wind(self.state)
        self.aero = Aerodynamics(self.state)
        self.thrust = Thrust(self.state)
        self.intg = get_integrator(self.state.timestep, self.eom)
        
    def eom(self, t, x, delta):
        # Update rigid body state
        self.state.rigid_body = x
        # Split up drone inputs delta
        delta_aero = delta[ :3] # elevator, aileron, rudder setting
        delta_thrust = delta[3] # thrust setting
        # Update wind
        self.wind.update()
        # Update aero force and moment
        self.aero.update(delta_aero) 
        # Update thrust force and moment
        self.thrust.update(delta_thrust) 
        # Add aero, thrust force and moment
        force_moment = self.aero.force_moment \
            + self.thrust.force_moment
        # Add gravity force
        force_moment[ :3] += self.gravity.force
        return self.rigid_body.eom(t, x, force_moment)
         
    def update(self, delta):
        x = self.intg.step(self.state.time, self.state.rigid_body, delta)
        self.state.time += self.state.timestep
        self.state.rigid_body = x 
        
        
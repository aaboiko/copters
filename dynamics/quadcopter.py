import numpy as np

from dynamics.dynamics import Dynamics


class Quadcopter(Dynamics):
    def __init__(self, pose, dpose, ddpose=np.zeros(3), mass=1.0):
        super().__init__(pose, dpose, ddpose=ddpose, mass=mass)
        self.type = "quadcopter"


    def step(self, dt, force, external_force=np.zeros(3)):
        self.ddpose = (force + external_force) / self.mass
        self.pose, self.dpose = self.rk4_step(dt)

        return (self.pose, self.dpose, self.ddpose)
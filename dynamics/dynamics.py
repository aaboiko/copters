import numpy as np


class Dynamics:
    def __init__(self, pose, dpose, ddpose=np.zeros(3), mass=1.0):
        self.mass = mass
        self.pose = pose
        self.dpose = dpose
        self.ddpose = ddpose


    def rk4_step(self, dt):
        k1_v = self.ddpose
        k2_v = self.ddpose  
        k3_v = self.ddpose
        k4_v = self.ddpose
        
        k1_p = self.dpose
        k2_p = self.dpose + 0.5 * dt * k1_v
        k3_p = self.dpose + 0.5 * dt * k2_v
        k4_p = self.dpose + dt * k3_v
        
        new_velocity = self.dpose + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        new_pose = self.pose + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        return (new_pose, new_velocity)
    

    def step(self, dt, force, external_force=np.zeros(3)):
        return (self.pose, self.dpose, self.ddpose)
import numpy as np

from sensors.transmitters import UpperTransmitter
from sensors.upper_receiver import UpperReceiver

motion_func_static_zero = lambda t: np.zeros(3)
motion_func_linear = lambda t: np.array([0.05, 0, 0]) * t
motion_func_rotation = lambda t: np.zeros(3), 

anchor_motions_functions = {
    "static_zero": motion_func_static_zero,
    "linear": motion_func_linear,
    "rotation": motion_func_rotation
}


class Anchor:
    def __init__(self, 
                 world,
                 motion_func,
                 n_agents,
                 pose=np.zeros(3), 
                 id=0,
                 agents_above=set(),
                 is_following_allowed=False,
                 t_start=0,
                 pattern=-1):
        
        self.world = world
        self.motion_func = motion_func
        self.n_agents = n_agents
        self.pose = pose
        self.id = id
        self.agents_above = agents_above
        self.t = t_start
        self.is_following_allowed = is_following_allowed
        self.pattern = pattern

        self.upper_transmitter = UpperTransmitter(world)
        self.upper_receiver = UpperReceiver(world)


    def allow_following(self):
        self.is_following_allowed = True


    def find_patterns(self):
        patterns = self.upper_receiver.request_signal(self.pose)

        for p in patterns:
            self.agents_above.add(p)

        if len(self.agents_above) == self.n_agents:
            self.allow_following()


    def broadcast(self):
        if self.is_following_allowed:
            self.upper_transmitter.signal_on(self.pattern, self.pattern, self.pose)
        else:
            self.upper_transmitter.signal_off(self.pattern)


    def step(self, dt):
        self.pose = self.motion_func(self.t)
        self.find_patterns()
        self.broadcast()
        self.world.logger.log_anchor_state(self)
        self.t += dt
        print(f"above: {self.agents_above}, is following allowed: {self.is_following_allowed}")

    
import numpy as np

from sensors.upper_receiver import UpperReceiver
from sensors.transmitters import UpperTransmitter, LowerTransmitter


class Agent:
    def __init__(
            self,
            world,
            sensor,
            dynamics,
            w,
            id=0, 
            t_start=0,
            mass=1.0,
            x=0,
            y=0,
            z=0,
            dx=0,
            dy=0,
            dz=0,
            is_active=True,
            is_awake=False, 
            u_max=1.0,
            v_bound=np.inf,
            rising_speed_bound=np.inf,
            is_alarmable=True):
        
        self.world = world
        self.sensor = sensor
        self.upper_receiver = UpperReceiver(world)
        self.upper_transmitter = UpperTransmitter(world)
        self.lower_transmitter = LowerTransmitter(world)
        self.dynamics = dynamics

        self.w = w
        self.id = id
        self.t = t_start
        self.mass = mass

        self.pose = np.array([x, y, z], dtype="f")
        self.dpose = np.array([dx, dy, dz], dtype="f")
        self.ddpose = np.zeros(3)
        self.force = np.zeros(3)

        self.is_active = is_active
        self.is_awake = is_awake
        self.u_max = u_max
        self.v_bound = v_bound
        self.rising_speed_bound = rising_speed_bound
        
        self.ordinary_peers = []
        self.anchor_peer = None
        self.signal_pattern = self.id
        self.is_alarmable = is_alarmable
        self.is_anchor_following_allowed = False


    def deactivate(self):
        self.is_active = False


    def activate(self):
        self.is_active = True


    def set_awake(self):
        self.is_awake = True
        self.upper_transmitter.signal_off(self.id)


    def g_vec(self, dists, gaps):
        return np.array([dist if not np.isinf(dist) else gap for dist, gap in zip(dists, gaps)])
    

    def find_peers(self):
        self.ordinary_peers, self.anchor_peer = self.sensor.get_peers_of_agent(self)

        if self.is_alarmable and self.anchor_peer is not None:
            self.lower_transmitter.signal_on(self.signal_pattern, self.id, self.pose)


    def find_patterns(self):
        pattern, self.is_anchor_following_allowed = self.sensor.get_pattern_to_repeat(self)
        
        if self.is_alarmable:
            if pattern is not None:
                self.lower_transmitter.signal_on(pattern, self.id, self.pose)
                self.is_alarmable = False
                
        else:
            self.lower_transmitter.signal_off(self.id)


    def upper_broadcast(self):
        self.upper_transmitter.signal_on(self.signal_pattern, self.id, self.pose)


    def upper_receive(self):
        patterns = self.upper_receiver.request_signal(self.pose)

        if self.signal_pattern in patterns:
            self.set_awake()


    def get_distances_from_vec(self, vecs):
        vecs = vecs if len(vecs.shape) == 2 else vecs.reshape((1, 3))
        vec = vecs[np.argmax(vecs[:, 2])]
        distances_plus = np.where(vec >= 0, vec, np.inf)
        distances_minus = np.abs(np.where(vec < 0, vec, -np.inf))

        #if self.id == 0:
        #    print(f"plus: {distances_plus}, minus: {distances_minus}")

        return (distances_plus, distances_minus)
    

    def force_bound_wrapper(self, force_raw, v_bound):
        v_cur = np.linalg.norm(self.dpose)
        is_force_along_velocity = np.dot(self.dpose, force_raw) >= 0
        is_speed_safe = v_cur <= v_bound
        is_acc_allowed = is_speed_safe or not is_force_along_velocity
        force_flag = float(is_acc_allowed)

        #print(f"v_cur: {v_cur}, is_along: {is_force_along_velocity}, is_safe: {is_speed_safe}, flag: {force_flag}")

        return self.u_max * force_flag * force_raw / np.linalg.norm(force_raw) if np.linalg.norm(force_raw) > 0 else \
             np.zeros(3)


    def calculate_forces(self, xi=lambda arg: np.tanh(arg)):
        consider_anchor = self.anchor_peer is not None and self.is_anchor_following_allowed
        peers = self.ordinary_peers + [self.anchor_peer] if consider_anchor else self.ordinary_peers

        if peers:
            vecs = np.array([p.tolist() for p in peers])
            ds_plus, ds_minus = self.get_distances_from_vec(vecs)
            gaps = np.array([0, 0, self.w])
            force_raw = -np.sign(self.dpose - xi(self.g_vec(ds_plus, gaps)) + xi(self.g_vec(ds_minus, gaps)))
            v_bound = self.v_bound
        else:
            force_raw = self.u_max * np.array([0, 0, 1])
            v_bound = self.rising_speed_bound

        self.force = self.force_bound_wrapper(force_raw, v_bound)


    def dynamics_step(self, dt):
        external_force = self.world.external_force_func(
            self.pose[0], 
            self.pose[1], 
            self.pose[2],
            self.t)

        self.pose, self.dpose, self.ddpose = \
            self.dynamics.step(dt, self.force, external_force=external_force)
        

    def step(self, dt):
        self.world.logger.log_agent_state(self)

        if self.is_active:
            if self.is_awake:
                self.find_peers()
                self.find_patterns()

                self.calculate_forces()
                self.dynamics_step(dt)
            else:
                self.upper_broadcast()
                self.upper_receive()

        self.t += dt
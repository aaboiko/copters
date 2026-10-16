import numpy as np

from agent.agent import Agent
from anchor import Anchor
from sensors.conic_perceptor import ConicPerceptor
from dynamics.double_integrator import DoubleIntegrator
from tqdm import tqdm


class World:
    def __init__(self,
                 scenario_name,
                 logger,
                 agents_initial_poses,
                 agents_initial_velocities,
                 agents_activity_mask,
                 activation_events,
                 external_force_func,
                 anchor_motion_func,
                 agents_dynamics,
                 camera_fov,
                 w,
                 u_max_all=1.0,
                 dt=0.01,
                 n_iters=10000,
                 has_anchor=False,
                 restore_prev_run_state=False,
                 v_bound=np.inf,
                 rising_speed_bound=np.inf):
        
        self.scenario_name = scenario_name
        self.logger = logger
        self.u_max_all = u_max_all
        self.dt = dt
        self.n_iters = n_iters
        self.w = w
        self.camera_fov = camera_fov
        self.has_anchor = has_anchor
        self.v_bound = v_bound
        self.rising_speed_bound = rising_speed_bound

        self.external_force_func = external_force_func if external_force_func is not None else lambda x, y, z, t: np.zeros(3)
        self.anchor_motion_func = anchor_motion_func
        self.agents_dynamics = agents_dynamics
        self.agents_activity_mask = agents_activity_mask 

        self.activation_events = {moment["time"]: ([(event["id"], event["type"]) for event in moment["events"]]) 
                                  for moment in activation_events} \
                                  if activation_events is not None else {}

        print("creating world...")

        if restore_prev_run_state:
            n_agents_saved = self.logger.get_n_agents()
            t_max = self.logger.get_t_max()
            n_iters = self.logger.get_n_iters()
            
            last_moment_states_agents = self.logger.get_momental_agents_states(t_max)
            self.n_agents = n_agents_saved
            self.time = t_max + self.dt

            self.agents_initial_poses = [
                np.array([
                    state["pose"]["x"], 
                    state["pose"]["y"],
                    state["pose"]["z"]
                ]) 
                for state in last_moment_states_agents
            ]

            self.agents_initial_velocities = [
                np.array([
                    state["dpose"]["x"], 
                    state["dpose"]["y"],
                    state["dpose"]["z"]
                ]) 
                for state in last_moment_states_agents
            ]

            self.agents_activity_mask = [state["is_active"] for state in last_moment_states_agents]
            self.agents_awakeness_mask = [state["is_awake"] for state in last_moment_states_agents]
            self.agents_alarmability_mask = [state["is_alarmable"] for state in last_moment_states_agents]

            last_moment_state_world = self.logger.get_momental_world_state(t_max)

            self.signals = {
                item["id"]: {
                    "pattern": item["pattern"],
                    "pose": item["pose"]
                }

                for item in last_moment_state_world["signals"]
            } 

        else:
            self.n_agents = len(agents_initial_poses)
            self.time = 0
            self.agents_initial_poses = agents_initial_poses
            self.agents_initial_velocities = agents_initial_velocities

            self.agents_awakeness_mask = [False if i > 0 else True for i, state in enumerate(agents_initial_poses)]
            self.agents_alarmability_mask = [True for _ in agents_initial_poses]
            self.signals = dict()

        self.agents = self.create_agents()
        self.anchor = self.create_anchor(restore_prev_run_state)

        self.run_completed = False
        print("world created")


    def create_dynamics(self, pose, dpose, mass=1.0):
        dynamics_dict = {
            "double_integrator": DoubleIntegrator(pose, dpose, mass=mass)
        }

        return dynamics_dict[self.agents_dynamics]

        
    def create_agents(self):
        return [
            Agent(self,
                  ConicPerceptor(self, self.camera_fov),
                  self.create_dynamics(pose, dpose),
                  self.w,
                  id=i,
                  t_start=self.time,
                  x=pose[0],
                  y=pose[1],
                  z=pose[2],
                  dx=dpose[0],
                  dy=dpose[1],
                  dz=dpose[2],
                  is_active=is_active,
                  is_awake=is_awake,
                  u_max=self.u_max_all,
                  is_alarmable=is_alarmable,
                  v_bound=self.v_bound,
                  rising_speed_bound=self.rising_speed_bound)

            for i, (
                pose, 
                dpose, 
                is_active,
                is_awake,
                is_alarmable
            ) 
            
            in enumerate(
                zip(
                    self.agents_initial_poses, 
                    self.agents_initial_velocities,
                    self.agents_activity_mask,
                    self.agents_awakeness_mask,
                    self.agents_alarmability_mask
                )
            )
        ]
    

    def create_anchor(self, restore):
        if restore:
            last_moment_state = self.logger.get_momental_anchor_state(self.time - self.dt)

            pose = np.array([last_moment_state["pose"]["x"], 
                            last_moment_state["pose"]["y"],
                            last_moment_state["pose"]["z"]])
            
            agents_above = set(last_moment_state["agents_above"])
            is_following_allowed = last_moment_state["is_following_allowed"]
        else:
            pose = self.anchor_motion_func(0)
            agents_above = set()
            is_following_allowed = False

        return Anchor(self,
                      self.anchor_motion_func,
                      self.n_agents,
                      pose=pose,
                      agents_above=agents_above,
                      is_following_allowed=is_following_allowed,
                      t_start=self.time)
    

    def step(self):
        if self.time in self.activation_events.keys():
            for agent_id, activation_state in self.activation_events[self.time]:
                if activation_state:
                    self.agents[agent_id].activate()
                else:
                    self.agents[agent_id].deactivate()

        for agent in self.agents:
            agent.step(self.dt)

        self.anchor.step(self.dt)

        self.logger.log_world_state(self)
        self.time += self.dt


    def run(self):
        print("running scenario...")

        for iter in tqdm(range(self.n_iters)):
            self.step()

        self.logger.finish()
        self.run_completed = True
        print("finished")


    def signal_on(self, pattern, id, pose):
        if pattern is not None:
            self.signals[id] = {
                "id": id,
                "pattern": pattern, 
                
                "pose": {
                    "x": pose[0],
                    "y": pose[1],
                    "z": pose[2]
                }
            }


    def signal_off(self, id):
        if self.signals.get(id) is not None:
            self.signals.pop(id, None)


    def request_signal(self):
        return list(self.signals.values())
import numpy as np

from sensors.sensor import Sensor


class ConicPerceptor(Sensor):
    def __init__(self, world, alpha, axis=np.array([0, 0, -1])):
        super().__init__(world)
        self.type = "conic_perceptor"
        self.axis = axis
        self.alpha = alpha


    def check_point_inside_cone(self, point, x0, axis, alpha):
        axis_norm = np.linalg.norm(axis)
        vec_to_point = point - x0
        vec_norm = np.linalg.norm(vec_to_point)

        return np.dot(vec_to_point, axis) / (vec_norm * axis_norm) >= np.cos(alpha / 2)


    def get_observable_peers(self, points, agent_pose):
        return [point for point in points if self.check_point_inside_cone(point, agent_pose, self.axis, self.alpha)]
    

    def get_visible_anchor_pose(self, cur_agent):
        anchor_pose = self.world.anchor.pose
        is_visible = self.check_point_inside_cone(anchor_pose, cur_agent.pose, self.axis, self.alpha)
        return anchor_pose - cur_agent.pose if is_visible else None


    def get_peers_of_agent(self, cur_agent):
        ordinary_peers = super().get_ordinary_peers_of_agent(cur_agent, self.get_observable_peers)
        anchor_peer = self.get_visible_anchor_pose(cur_agent)

        return (ordinary_peers, anchor_peer)
    

    def pose_dict_to_numpy(self, pose_dict):
        return np.array([
            pose_dict["x"],
            pose_dict["y"],
            pose_dict["z"]
        ])
    

    def get_distance_to_signal(self, signal_pose, pose):
        return np.linalg.norm(self.pose_dict_to_numpy(signal_pose) - pose)
    

    def get_observable_patterns(self, cur_agent, signals):
        return [
            s for s in signals
            if self.get_distance_to_signal(s["pose"], cur_agent.pose) > 0.001 and
            self.check_point_inside_cone(self.pose_dict_to_numpy(s["pose"]), cur_agent.pose, self.axis, self.alpha) 
        ]
    

    def get_pattern_to_repeat(self, cur_agent):
        signals = self.world.request_signal()
        obs_signals = self.get_observable_patterns(cur_agent, signals)
        obs_signals.sort(key=lambda s: self.get_distance_to_signal(s["pose"], cur_agent.pose))

        to_repeat = obs_signals[0]["pattern"] if obs_signals else None
        has_anchor = -1 in {s["pattern"] for s in obs_signals}
        return (to_repeat, has_anchor)

import numpy as np


class UpperReceiver:
    def __init__(self, world):
        self.world = world


    def pose_dict_to_numpy(self, pose_dict):
        return np.array([
            pose_dict["x"],
            pose_dict["y"],
            pose_dict["z"]
        ])


    def check_is_visible(self, signal_pose, cur_agent_pose):
        return signal_pose[2] > cur_agent_pose[2]


    def request_signal(self, cur_agent_pose):
        signals = self.world.request_signal()
        
        return [s["pattern"] for s in signals 
                if self.check_is_visible(self.pose_dict_to_numpy(s["pose"]), cur_agent_pose)]
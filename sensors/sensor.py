import numpy as np


class Sensor:
    def __init__(self, world):
        self.world = world
        self.type = None

    def get_data(self, cur_agent):
        pass


    def get_peers_of_agent(self, cur_agent, get_observable_peers_func):
        points_list = [agent.pose.tolist() 
                        for agent in self.world.agents 
                        if agent.id != cur_agent.id and agent.is_active]

        if self.world.n_agents < 2 or not points_list:
            return [self.world.anchor_pose - cur_agent.pose]
        
        points = np.array(points_list)
        points = np.vstack([points, self.world.anchor.pose]) if self.world.has_anchor else points
        peers_poses = get_observable_peers_func(points, cur_agent.pose)

        return [peer_pose - cur_agent.pose for peer_pose in peers_poses]
    

    def get_ordinary_peers_of_agent(self, cur_agent, get_observable_peers_func):
        points_list = [agent.pose.tolist() 
                        for agent in self.world.agents 
                        if agent.id != cur_agent.id and agent.is_active]
        
        points = np.array(points_list)
        peers_poses = get_observable_peers_func(points, cur_agent.pose)

        return [peer_pose - cur_agent.pose for peer_pose in peers_poses]
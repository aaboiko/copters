import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Circle
from tqdm import tqdm
from functools import partial


class Visualizer:
    def __init__(self,
                 scenario_name,
                 logger,
                 agents_initialization_bounds,
                 camera_fov,
                 has_anchor=False,
                 fontsize_label=30,
                 font_size_tick=26,
                 s_point=120,
                 linewidth=5,
                 labels_fontsize=20,
                 gif_frame_step=25,
                 gif_n_frames=150):
        
        self.scenario_name = scenario_name
        self.logger = logger
        self.camera_fov = camera_fov
        self.has_anchor = has_anchor
        self.agents_initialization_bounds = agents_initialization_bounds

        self.fontsize_label = fontsize_label
        self.font_size_tick = font_size_tick
        self.s_point = s_point
        self.s_anchor = 2 * self.s_point
        self.linewidth = linewidth
        self.labels_fontsize = labels_fontsize
        self.gif_frame_step = gif_frame_step
        self.gif_n_frames = gif_n_frames

        x_bounds, y_bounds, z_bounds = self.agents_initialization_bounds
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        z_min, z_max = z_bounds

        self.x_min = x_min - 1
        self.y_min = y_min - 1
        self.z_min = z_min

        self.x_max = x_max + 1
        self.y_max = y_max + 1
        self.z_max = z_max

        self.n_agents = None
        self.n_iters = None
        self.t_max = None

        self.ts = None
        self.syncronized = False

        self.fig, self.ax = plt.subplots(figsize=(40, 40))
        self.fig_xy, self.ax_xy = plt.subplots(figsize=(40, 40))
        self.fig_xz, self.ax_xz = plt.subplots(figsize=(40, 40))
        self.fig_3d, self.ax_3d = plt.subplots(figsize=(40, 40), subplot_kw={'projection': '3d'})
        self.circle_alpha = 1.0

        self.info_dict = {
            "xy": " in xy projection",
            "xz": " in xz projection",
            "3d": " in 3d projection"
        }

        self.fig_dict = {
            "xy": self.fig_xy,
            "xz": self.fig_xz,
            "3d": self.fig_3d
        }

        self.axes_dict = {
            "xy": self.ax_xy,
            "xz": self.ax_xz,
            "3d": self.ax_3d
        }


    def sync_with_database(self):
        self.n_agents = self.logger.get_n_agents()
        self.t_max = self.logger.get_t_max()
        self.n_iters = self.logger.get_n_iters()
        self.ts = self.logger.get_ts()
        
        self.syncronized = True
        self.circle_alpha = 1.0 / self.n_agents
        self.gif_frame_step = self.n_iters // self.gif_n_frames


    def adjust_ax_xy(self):
        self.ax_xy.set_xlim((self.x_min, self.x_max))
        self.ax_xy.set_ylim((self.y_min, self.y_max))
        self.ax_xy.set_aspect('equal', adjustable='box')
        
        self.ax_xy.set_xlabel('x', fontsize=self.fontsize_label)
        self.ax_xy.set_ylabel('y', fontsize=self.fontsize_label)

        self.ax_xy.tick_params(axis="x", labelsize=self.font_size_tick)
        self.ax_xy.tick_params(axis="y", labelsize=self.font_size_tick)


    def adjust_ax_xz(self):
        self.ax_xz.set_xlim((self.x_min, self.x_max))
        self.ax_xz.set_ylim((self.z_min, self.z_max))
        self.ax_xz.set_aspect('equal', adjustable='box')
        
        self.ax_xz.set_xlabel('x', fontsize=self.fontsize_label)
        self.ax_xz.set_ylabel('z', fontsize=self.fontsize_label)

        self.ax_xz.tick_params(axis="x", labelsize=self.font_size_tick)
        self.ax_xz.tick_params(axis="y", labelsize=self.font_size_tick)


    def adjust_ax_3d(self):
        self.ax_3d.set_xlim((self.x_min, self.x_max))
        self.ax_3d.set_ylim((self.y_min, self.y_max))
        self.ax_3d.set_zlim((self.z_min, self.z_max))

        self.ax_3d.set_box_aspect(
        (   
                self.x_max - self.x_min, 
                self.y_max - self.y_min, 
                self.z_max - self.z_min,
            )
        )

        self.ax_3d.set_xlabel('x', fontsize=self.fontsize_label)
        self.ax_3d.set_ylabel('y', fontsize=self.fontsize_label)
        self.ax_3d.set_zlabel('z', fontsize=self.fontsize_label)

        self.ax_3d.tick_params(axis="x", labelsize=self.font_size_tick)
        self.ax_3d.tick_params(axis="y", labelsize=self.font_size_tick)
        self.ax_3d.tick_params(axis="z", labelsize=self.font_size_tick)


    def adjust_ax(self, axes="xy"):
        adjust_ax_funcs_dict = {
            "xy": self.adjust_ax_xy,
            "xz": self.adjust_ax_xz,
            "3d": self.adjust_ax_3d
        }

        adjust_ax_funcs_dict[axes]()


    def plot_anchor_xy(self, anchor_pose):
        x, y, z = anchor_pose
        self.ax_xy.scatter(x, y, s=self.s_anchor, color="red")


    def plot_anchor_xz(self, anchor_pose):
        x, y, z = anchor_pose
        self.ax_xz.scatter(x, z, s=self.s_anchor, color="red")


    def plot_anchor_3d(self, anchor_pose):
        x, y, z = anchor_pose
        self.ax_3d.scatter(x, y, z, s=self.s_anchor, color="red")
            
    
    def plot_anchor(self, anchor_data, axes="xy"):
        plot_anchor_funcs_dict = {
            "xy": self.plot_anchor_xy,
            "xz": self.plot_anchor_xz,
            "3d": self.plot_anchor_3d
        }

        anchor_pose = np.array([anchor_data["pose"]["x"], 
                                anchor_data["pose"]["y"],
                                anchor_data["pose"]["z"]])

        plot_anchor_funcs_dict[axes](anchor_pose)


    def plot_agent_xy(self,
                      pose,
                      agent_id,
                      is_awake,
                      show_agent_id=False):
        
        agent_x, agent_y, agent_z = pose

        self.ax_xy.scatter(agent_x, 
                            agent_y, 
                            s=self.s_point*2, 
                            color="green" if is_awake else "blue")
        
        R_vis = agent_z * np.tan(self.camera_fov / 2)
        circle = Circle(xy=(agent_x, agent_y), radius=R_vis, color="grey", alpha=self.circle_alpha)
        self.ax_xy.add_artist(circle)

        if show_agent_id:
            self.ax_xy.text(agent_x - 0.04, agent_y - 0.04, f"{agent_id}", fontsize=self.labels_fontsize, ha="center")


    def plot_agent_xz(self,
                      pose,
                      agent_id,
                      is_awake,
                      show_agent_id=False):
        
        agent_x, agent_y, agent_z = pose

        self.ax_xz.scatter(agent_x, 
                            agent_z, 
                            s=self.s_point*2, 
                            color="green" if is_awake else "blue")
        
        R_vis = agent_z * np.tan(self.camera_fov / 2)

        poly_xs = [
            agent_x - R_vis,
            agent_x,
            agent_x + R_vis
        ]

        poly_zs = [
            self.z_min,
            agent_z,
            self.z_min
        ]

        self.ax_xz.fill(poly_xs, poly_zs, color='grey', edgecolor='grey', alpha=self.circle_alpha)

        if show_agent_id:
            self.ax_xz.text(agent_x - 0.04, agent_z - 0.04, f"{agent_id}", fontsize=self.labels_fontsize, ha="center")


    def plot_agent_3d(self,
                      pose,
                      agent_id,
                      is_awake,
                      show_agent_id=False):
        
        agent_x, agent_y, agent_z = pose

        self.ax_3d.scatter(agent_x, 
                        agent_y, 
                        agent_z,
                        s=self.s_point*2, 
                        color="green" if is_awake else "blue")


    def plot_agent(self,
                   pose,
                   agent_id,
                   is_awake,
                   show_agent_id=False,
                   axes="xy"):
        
        plot_agents_funcs_dict = {
            "xy": self.plot_agent_xy,
            "xz": self.plot_agent_xz,
            "3d": self.plot_agent_3d
        }

        plot_agents_funcs_dict[axes](pose, agent_id, is_awake, show_agent_id=show_agent_id)


    def clear_scene(self, axes="xy"):
        ax_dict = {
            "xy": self.ax_xy,
            "xz": self.ax_xz,
            "3d": self.ax_3d
        }

        ax_dict[axes].clear()


    def draw_scene(self,
                   anchor_data,
                   agents_to_project=[],
                   show_agents_ids=False,
                   json_agents=True,
                   axes="xy"):
        
        self.clear_scene(axes=axes)
        self.adjust_ax(axes=axes)

        for agent in agents_to_project:
            is_active = agent["is_active"] if json_agents else agent.is_active

            if is_active:
                is_awake = agent["is_awake"] if json_agents else agent.is_awake
                agent_id = agent["id"] if json_agents else agent.id

                pose = np.array([
                    agent["pose"]["x"], 
                    agent["pose"]["y"],
                    agent["pose"]["z"]
                ]) if json_agents else agent.pose
                
                self.plot_agent(pose,
                                agent_id,
                                is_awake,
                                show_agent_id=show_agents_ids,
                                axes=axes)
                
        if self.has_anchor:
            self.plot_anchor(anchor_data, axes=axes)


    def plot_state(self, state="final", axes="xy"):
        state_iter_dict = {
            "initial": 0,
            "middle": self.n_iters // 2,
            "final": -1
        }

        info = self.info_dict[axes]
        print(f'plotting agents {state} positions{info}...')
        path = f"plots/{state}_positions/{self.scenario_name}_{axes}.jpg"
        i = state_iter_dict[state]

        agents_to_project = self.logger.get_momental_agents_states(self.ts[i])
        anchor_data = self.logger.get_momental_anchor_state(self.ts[i])

        self.draw_scene(anchor_data,
                        agents_to_project=agents_to_project,
                        show_agents_ids=True,
                        axes=axes)
        
        self.fig_dict[axes].savefig(path)


    def plot_final_positions(self, axes="xy"):
        self.plot_state(state="final", axes=axes)


    def plot_middle_positions(self, axes="xy"):
        self.plot_state(state="middle", axes=axes)


    def plot_initial_positions(self, axes="xy"):
        self.plot_state(state="initial", axes=axes)


    def animate(self,
                i,
                show_agents_ids=False,
                axes="xy"):
        
        agents_to_project = self.logger.get_momental_agents_states(self.ts[i])
        anchor_data = self.logger.get_momental_anchor_state(self.ts[i])

        self.draw_scene(anchor_data,
                        agents_to_project=agents_to_project,
                        show_agents_ids=show_agents_ids,
                        axes=axes)
        

    def create_gif(self, axes="xy"):
        info = self.info_dict[axes]
        print(f'creating gif{info}...')
        path = f"gifs/{axes}/{self.scenario_name}_{axes}.gif"

        anim = animation.FuncAnimation(self.fig_dict[axes], 
                                       partial(self.animate, 
                                               show_agents_ids=False,
                                               axes=axes), 
                                       frames = tqdm(range(0, self.n_iters, self.gif_frame_step)), 
                                       interval = 20)
        
        anim.save(path, fps = 60, writer = 'pillow')


    def plot_trajectories_world(self, axes="xy"):
        print(f'plotting agents global trajectories{self.info_dict[axes]}...')
        path = f"plots/trajectories/{self.scenario_name}_{axes}.jpg"
        ax = self.axes_dict[axes]
        ax.clear()

        x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
        x_min, y_min, z_min = np.inf, np.inf, np.inf

        for i in tqdm(range(self.n_agents)):
            start_state = self.logger.get_certain_agent_momental_state(i, self.ts[0])
            x_prev = start_state["pose"]["x"]
            y_prev = start_state["pose"]["y"]
            z_prev = start_state["pose"]["z"]
            color = "blue"
            
            for t in self.ts[1::10]:
                state = self.logger.get_certain_agent_momental_state(i, t)
                is_active = state["is_active"]
                x = state["pose"]["x"]
                y = state["pose"]["y"]
                z = state["pose"]["z"]

                if is_active:
                    val_2_dict = {
                        "xy": [y_prev, y],
                        "xz": [z_prev, z]
                    }

                    ax.plot([x_prev, x], val_2_dict[axes], color=color)

                x_prev = x
                y_prev = y
                z_prev = z

                x_max = max(x_max, x)
                y_max = max(y_max, y)
                z_max = max(z_max, z)

                x_min = min(x_min, x)
                y_min = min(y_min, y)
                z_min = min(z_min, z)

        ax.set_xlabel('x', fontsize=self.fontsize_label)
        ax.set_ylabel(axes[1], fontsize=self.fontsize_label)

        ax.tick_params(axis="x", labelsize=self.font_size_tick)
        ax.tick_params(axis="y", labelsize=self.font_size_tick)

        y_lim_dict = {
            "xy": (y_min - 0.1, y_max + 0.1),
            "xz": (z_min - 0.1, z_max + 0.1)
        }

        ax.set_xlim((x_min - 0.1, x_max + 0.1))
        ax.set_ylim(y_lim_dict[axes])

        self.fig_dict[axes].savefig(path, bbox_inches='tight', pad_inches=0)


    def plot_convergence(self):
        print('plotting agents convergence...')
        path = f"plots/convergence/{self.scenario_name}.jpg"
        self.ax.clear()

        d_min, d_max = -1, -np.inf

        for i in tqdm(range(self.n_agents)):
            agents_states_start = self.logger.get_momental_agents_states(self.ts[0])
            anchor_data_start = self.logger.get_momental_anchor_state(self.ts[i])

            anchor_pose_start = np.array([anchor_data_start["pose"]["x"], 
                                anchor_data_start["pose"]["y"],
                                anchor_data_start["pose"]["z"]])

            pose_ref_start = anchor_pose_start[0:2] if self.has_anchor else np.mean(np.array([
                [
                    state["pose"]["x"], 
                    state["pose"]["y"],
                    state["pose"]["z"]
                ]

                for state in agents_states_start

            ])[:, 0:2], axis=0)

            start_state = self.logger.get_certain_agent_momental_state(i, self.ts[0])

            start_pose = np.array([
                start_state["pose"]["x"],
                start_state["pose"]["y"]
            ])

            d_prev = np.linalg.norm(pose_ref_start - start_pose)
            t_prev = self.ts[0]
            color = "blue"

            for t in self.ts[1::10]:
                agents_states = self.logger.get_momental_agents_states(t)
                anchor_data = self.logger.get_momental_anchor_state(self.ts[i])

                anchor_pose = np.array([anchor_data["pose"]["x"], 
                                    anchor_data["pose"]["y"],
                                    anchor_data["pose"]["z"]])

                pose_ref = anchor_pose[0:2] if self.has_anchor else np.mean(np.array([
                    [
                        state["pose"]["x"], 
                        state["pose"]["y"],
                        state["pose"]["z"]
                    ]

                    for state in agents_states

                ])[:, 0:2], axis=0)

                state = self.logger.get_certain_agent_momental_state(i, t)
                is_active = state["is_active"]

                pose = np.array([
                    state["pose"]["x"],
                    state["pose"]["y"]
                ])

                d = np.linalg.norm(pose_ref - pose)

                if is_active:
                    self.ax.plot([t_prev, t], [d_prev, d], color=color)

                d_prev = d
                t_prev = t
                d_max = max(d_max, d)

        self.ax.set_xlabel('t, s', fontsize=self.fontsize_label)
        self.ax.set_ylabel('d, m', fontsize=self.fontsize_label)
        self.ax.tick_params(axis="x", labelsize=self.font_size_tick)
        self.ax.tick_params(axis="y", labelsize=self.font_size_tick)

        self.ax.set_xlim((0, self.ts[-1]))
        self.ax.set_ylim((d_min - 0.1, d_max + 0.1))

        self.fig.savefig(path, bbox_inches='tight', pad_inches=0)


    def check_collisions(self):
        print("checking collisions throughout logdata...")
        n = self.n_agents + 1
        proximity_matrix = np.inf * np.ones((n, n))
        
        for t in tqdm(self.ts):
            agents_data = self.logger.get_momental_agents_states(t)

            agents_poses = np.array([
                [
                    agent_data["pose"]["x"],
                    agent_data["pose"]["y"],
                    agent_data["pose"]["z"],
                ]

                for agent_data in agents_data
            ])

            poses = np.vstack((agents_poses, self.anchor_pose))

            for i in range(n):
                cur_pose = poses[i, :]

                for j in range(n):
                    if i == j:
                        continue

                    another_pose = poses[j, :]
                    dist = np.linalg.norm(cur_pose - another_pose)
                    prev_dist = proximity_matrix[i, j]
                    proximity_matrix[i, j] = min(prev_dist, dist)

        min_dist = np.min(proximity_matrix)
        min_indices = np.unravel_index(np.argmin(proximity_matrix), proximity_matrix.shape)

        print(f"proximity matrix: {str(proximity_matrix)}")
        print(f"minimal distance: {min_dist}")
        print(f"indices: {int(min_indices[0])}, {int(min_indices[1])}")
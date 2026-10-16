import argparse
import yaml
import numpy as np

from external_force_functions import external_force_functions
from anchor import anchor_motions_functions
from logger import Logger, clear_all
from world import World
from visualizer import Visualizer


parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()
config_file_name = args.config
config_file_path = f"configs/{config_file_name}.yaml"

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

scenario_name = config["scenario_name"]
agents_initialization = config["agents_initialization"]

agents_initialization_bounds_obj = config["agents_initialization_bounds"]
agents_initialization_bounds = (
    (
        agents_initialization_bounds_obj["x_min"],
        agents_initialization_bounds_obj["x_max"]
    ),
    (
        agents_initialization_bounds_obj["y_min"],
        agents_initialization_bounds_obj["y_max"]
    ),
    (
        agents_initialization_bounds_obj["z_min"],
        agents_initialization_bounds_obj["z_max"]
    )
)

if agents_initialization == "random":
    x_bounds, y_bounds, z_bounds = agents_initialization_bounds
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds

    n_agents = config["n_agents"]

    agents_initial_poses = [
        np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            0
        ])
        for _ in range(n_agents)
    ]
else:
    init_file_path = f"initialization/{agents_initialization}.txt"
    poses_xy = np.loadtxt(init_file_path)
    n_agents = poses_xy.shape[0]
    agents_initial_poses = [np.array([x, y, 0]) for x, y in poses_xy]

agents_initial_velocities = [np.zeros(3) for _ in range(n_agents)]
agents_activity_mask_obj = config["agents_activity_mask"]

agents_activity_mask = [
    agents_activity_mask_obj[i] 
    if (i < len(agents_initial_velocities)) else True
    for i in range(n_agents)
] if agents_activity_mask_obj is not None else \
[True for _ in range(n_agents)]

has_anchor = config["has_anchor"]

camera_fov = np.deg2rad(config["camera_fov"])
w = config["w"]
u_max_all = config["u_max_all"]
dt = config["dt"]
n_iters = config["n_iters"]
v_bound = config["v_bound"]
rising_speed_bound = config["rising_speed_bound"]

activation_events = config["activation_events"]
external_force_func = external_force_functions[config["external_force_func"]]
anchor_motion_func = anchor_motions_functions[config["anchor_motion_func"]]
agent_dynamics = config["agent_dynamics"]

restore_prev_run_state = config["restore_prev_run_state"]
flag_run = config["flag_run"]
flag_create_gif_xy = config["flag_create_gif_xy"]
flag_create_gif_xz = config["flag_create_gif_xz"]
flag_create_gif_3d = config["flag_create_gif_3d"]
flag_plot_trajectories = config["flag_plot_trajectories"]
flag_plot_convergence = config["flag_plot_convergence"]
flag_check_collisions = config["flag_check_collisions"]
flag_plot_initial_and_middle_states = config["flag_plot_initial_and_middle_states"]
flag_plot_final_state_3d = config["flag_plot_final_state_3d"]
flag_plot_initial_and_middle_states_3d = config["flag_plot_initial_and_middle_states_3d"]
flag_clear_all = config["flag_clear_all"]

can_visualize = flag_run or (not flag_clear_all or restore_prev_run_state)

if flag_clear_all:
    clear_all()

logger = Logger(config)

visualizer = Visualizer(scenario_name,
                        logger,
                        agents_initialization_bounds,
                        camera_fov,
                        has_anchor=has_anchor)

if not restore_prev_run_state and flag_run and not flag_clear_all:
    logger.clear_scenario()

if flag_run:
    world = World(scenario_name,
                  logger,
                  agents_initial_poses,
                  agents_initial_velocities,
                  agents_activity_mask,
                  activation_events,
                  external_force_func,
                  anchor_motion_func,
                  agent_dynamics,
                  camera_fov,
                  w,
                  u_max_all=u_max_all,
                  dt=dt,
                  n_iters=n_iters,
                  has_anchor=has_anchor,
                  restore_prev_run_state=restore_prev_run_state,
                  v_bound=v_bound,
                  rising_speed_bound=rising_speed_bound)
    
    world.run()

if can_visualize:
    visualizer.sync_with_database()
    print(f"number of saved iters: {visualizer.n_iters}")

    visualizer.plot_final_positions()
    visualizer.plot_final_positions(axes="xz")

    if flag_plot_initial_and_middle_states:
        visualizer.plot_initial_positions()
        visualizer.plot_middle_positions()
        visualizer.plot_initial_positions(axes="xz")
        visualizer.plot_middle_positions(axes="xz")

    if flag_plot_final_state_3d:
        visualizer.plot_final_positions(axes="3d")

    if flag_plot_initial_and_middle_states_3d:
        visualizer.plot_initial_positions(axes="3d")
        visualizer.plot_middle_positions(axes="3d")

    if flag_plot_convergence:
        visualizer.plot_convergence()

    if flag_plot_trajectories:
        visualizer.plot_trajectories_world(axes="xz")

    if flag_create_gif_xy:
        visualizer.create_gif(axes="xy")
    if flag_create_gif_xz:
        visualizer.create_gif(axes="xz")
    if flag_create_gif_3d:
        visualizer.create_gif(axes="3d")

    if flag_check_collisions:
        visualizer.check_collisions()
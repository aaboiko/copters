import duckdb
import json
import uuid
import os

from pathlib import Path


class Logger:
    def __init__(self, config):
        self.config = config
        self.db_path = 'logs.duckdb'
        self.scenario_name = config["scenario_name"]
        self.run_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, self.scenario_name))

        self.batch_agents = []
        self.batch_anchors = []
        self.batch_world = []
        self.batch_size = config["log_batch_size"]
        self.agent_dynamics = config["agent_dynamics"]

        self.conn = duckdb.connect(self.db_path)
        self.create_metadata_table()
        self.initialize_scenario()


    def create_metadata_table(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id VARCHAR PRIMARY KEY,
                scenario_name VARCHAR NOT NULL UNIQUE,
                parameters JSON
            )
        """)


    def initialize_scenario(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                run_id VARCHAR,
                scenario_name VARCHAR,
                timestep FLOAT,
                agent_id VARCHAR,
                data JSON,
                PRIMARY KEY (run_id, timestep, agent_id),
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS anchors (
                run_id VARCHAR,
                scenario_name VARCHAR,
                timestep FLOAT,
                anchor_id VARCHAR,
                data JSON,
                PRIMARY KEY (run_id, timestep, anchor_id),
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS world (
                run_id VARCHAR,
                scenario_name VARCHAR,
                timestep FLOAT,
                data JSON,
                PRIMARY KEY (run_id, timestep),
                FOREIGN KEY (run_id) REFERENCES simulation_runs(run_id)
            )
        """)

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_run ON agents(run_id, timestep)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_anchor_run ON anchors(run_id, timestep)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obstacle_run ON world(run_id, timestep)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scenario_name ON simulation_runs(scenario_name)")

        simulation_runs_df = self.conn.sql("SELECT * FROM simulation_runs").df()

        if not self.scenario_name in simulation_runs_df["scenario_name"].values:
            self.conn.execute("""
                INSERT INTO simulation_runs 
                (run_id, scenario_name, parameters)
                VALUES (?, ?, ?)
            """, [
                self.run_id,
                self.scenario_name,
                json.dumps(self.config) if self.config else None
            ])

            self.conn.commit()


    def get_state_json_copter(self, agent):
        return {} if agent.dynamics.type == "quadcopter" else {}


    def get_state_json_common(self, agent):
        return {
            "id": agent.id,
            "mass": agent.mass,
            "u_max": agent.u_max,

            "pose": {
                "x": float(agent.pose[0]),
                "y": float(agent.pose[1]),
                "z": float(agent.pose[2])
            },

            "dpose": {
                "x": float(agent.dpose[0]),
                "y": float(agent.dpose[1]),
                "z": float(agent.dpose[2])
            },

            "ddpose": {
                "x": float(agent.ddpose[0]),
                "y": float(agent.ddpose[1]),
                "z": float(agent.ddpose[2])
            },

            "force": {
                "x": float(agent.force[0]),
                "y": float(agent.force[1]),
                "z": float(agent.force[2])
            },

            "is_active": agent.is_active,
            "is_awake": agent.is_awake,
            "is_alarmable": agent.is_alarmable
        }
    

    def get_state_json_anchor(self, anchor):
        return {
            "id": anchor.id,
            "agents_above": list(anchor.agents_above),
            "is_following_allowed": anchor.is_following_allowed,

            "pose": {
                "x": float(anchor.pose[0]),
                "y": float(anchor.pose[1]),
                "z": float(anchor.pose[2])
            },
        }
    

    def get_state_json_world(self, world):
        signals = [
            {
                "id": int(id),
                "pattern": int(value["pattern"]),

                "pose": {
                    "x": float(value["pose"]["x"]),
                    "y": float(value["pose"]["y"]),
                    "z": float(value["pose"]["z"])
                }
            }

            for id, value in world.signals.items()
        ]

        return {
            "signals": signals
        }
    

    def get_state_json_agent(self, agent):
        state_common = self.get_state_json_common(agent)

        states_dict = {
            "double_integrator": state_common,
            "quadcopter": state_common | self.get_state_json_copter(agent)
        }

        return states_dict[self.agent_dynamics]
    

    def flush_batch_agents(self):
        self.conn.executemany("""
            INSERT INTO agents (run_id, scenario_name, timestep, agent_id, data)
            VALUES (?, ?, ?, ?, ?)
        """, self.batch_agents)

        self.batch_agents.clear()


    def flush_batch_anchors(self):
        self.conn.executemany("""
            INSERT INTO anchors (run_id, scenario_name, timestep, anchor_id, data)
            VALUES (?, ?, ?, ?, ?)
        """, self.batch_anchors)

        self.batch_anchors.clear()


    def flush_batch_world(self):
        self.conn.executemany("""
            INSERT INTO world (run_id, scenario_name, timestep, data)
            VALUES (?, ?, ?, ?)
        """, self.batch_world)

        self.batch_world.clear()


    def log_agent_state(self, agent):
        state_json = self.get_state_json_agent(agent)
        self.batch_agents.append([self.run_id, self.scenario_name, agent.t, agent.id, state_json])
        
        if len(self.batch_agents) >= self.batch_size:
            self.flush_batch_agents()


    def log_anchor_state(self, anchor):
        state_json = self.get_state_json_anchor(anchor)
        self.batch_anchors.append([self.run_id, self.scenario_name, anchor.t, anchor.id, state_json])
        
        if len(self.batch_anchors) >= self.batch_size:
            self.flush_batch_anchors()


    def log_world_state(self, world):
        state_json = self.get_state_json_world(world)
        self.batch_world.append([self.run_id, self.scenario_name, world.time, state_json])
        
        if len(self.batch_world) >= self.batch_size:
            self.flush_batch_world()


    def get_n_agents(self):
        return self.conn.sql(f"SELECT \
                     count(DISTINCT agent_id) \
                     FROM agents \
                     where scenario_name = \'{self.scenario_name}\'").fetchone()[0]
    

    def get_t_max(self):
        return self.conn.sql(f"SELECT \
                     max(timestep) \
                     FROM agents \
                     where scenario_name = \'{self.scenario_name}\'").fetchone()[0]
    

    def get_ts(self):
        return self.conn.sql(f"SELECT \
                timestep FROM agents \
                where scenario_name = \'{self.scenario_name}\'").df().drop_duplicates()["timestep"].values


    def get_n_iters(self):
        n_iters =  self.conn.sql(f"SELECT count(*) \
                             FROM agents \
                             where scenario_name = \'{self.scenario_name}\' \
                             and agent_id = 0").fetchone()[0]
        
        if n_iters == 0:
            print("cannot obtain data from logs: database is empty")
            exit()
        
        return n_iters
    

    def get_momental_agents_states(self, timestep):
        n_agents = self.get_n_agents()

        df_agents = [
            self.conn.sql(f"SELECT \
                    data FROM agents \
                    where scenario_name = \'{self.scenario_name}\' \
                    and timestep = CAST({timestep} AS VARCHAR) \
                    and agent_id = {i}").df()["data"]
            
            for i in range(n_agents)
        ]

        #print(df_agents[2])

        return [json.loads(df.values[0]) for df in df_agents]
    

    def get_momental_anchor_state(self, timestep):
        df_anchor = self.conn.sql(f"SELECT \
                            data FROM anchors \
                            where scenario_name = \'{self.scenario_name}\' \
                            and anchor_id = 0 \
                            and timestep = CAST({timestep} AS VARCHAR)").df()["data"]
        
        return json.loads(df_anchor.values[0])
    

    def get_momental_world_state(self, timestep):
        df_world = self.conn.sql(f"SELECT \
                            data FROM world \
                            where scenario_name = \'{self.scenario_name}\' \
                            and timestep = CAST({timestep} AS VARCHAR)").df()["data"]
        
        return json.loads(df_world.values[0])
    

    def get_certain_agent_momental_state(self, agent_id, timestep):
        df_agent = self.conn.sql(f"SELECT \
                            data FROM agents \
                            where scenario_name = \'{self.scenario_name}\' \
                            and agent_id = {agent_id} \
                            and timestep = CAST({timestep} AS VARCHAR)").df()["data"]
        
        return json.loads(df_agent.values[0])
    

    def clear_scenario(self):
        scene_exists = self.conn.execute("""
            SELECT COUNT(*) FROM simulation_runs 
            WHERE scenario_name = ?
        """, [self.scenario_name]
        ).fetchone()[0] > 0
        
        if scene_exists:
            print("clearing the existing logs...")

            self.conn.execute("""
                DELETE FROM agents 
                WHERE scenario_name = ?
            """, [self.scenario_name]
            )

            self.conn.execute("""
                DELETE FROM anchors 
                WHERE scenario_name = ?
            """, [self.scenario_name]
            )

            self.conn.execute("""
                DELETE FROM world 
                WHERE scenario_name = ?
            """, [self.scenario_name]
            )

            self.conn.commit()
            print("finished")
        else:
            print(f"the scenario {self.scenario_name} is being runned first time")


    def finish(self):
        if len(self.batch_agents) > 0:
            self.flush_batch_agents()

        if len(self.batch_anchors) > 0:
            self.flush_batch_anchors()

        if len(self.batch_world) > 0:
            self.flush_batch_world()

        self.conn.commit()


    def close_connection(self):
        self.conn.close()


def clear_all():
    print("clearing all database...")
    path = Path('logs.duckdb')

    if path.exists():
        os.remove(path)

    print("database cleared")
import os
import traci
import numpy as np
import random

class SUMOEnv:
    def __init__(self, sumo_cfg='highway.sumocfg', max_steps=1000, gui=True):
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.gui = gui
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"
        self.ego_id = "ego"
        self.actions = ["keep_lane", "change_left", "change_right", "brake", "accelerate"]
        self.action_space = len(self.actions)
        self.step_num = 0
        self.cutin_spawned = False

    def reset(self):
        if traci.isLoaded():
            traci.close(False)

        sumo_cmd = [self.sumo_binary, "-c", self.sumo_cfg, "--start"]
        traci.start(sumo_cmd)
        self.step_num = 0
        self.cutin_spawned = False

        # Letting background traffic go first
        delay_steps = random.randint(100, 200)
        for _ in range(delay_steps):
            traci.simulationStep()
            self.step_num += 1

        # Waiting for ego to spawn and track it
        ego_tracked = False
        while not ego_tracked:
            traci.simulationStep()
            self.step_num += 1
            if self.ego_id in traci.vehicle.getIDList():
                try:
                    if self.gui:
                        traci.gui.trackVehicle("View #0", self.ego_id)
                        traci.gui.setZoom("View #0", 120)
                    ego_tracked = True
                except traci.TraCIException:
                    pass

        # Seting ego properties
        try:
            traci.vehicle.setSpeed(self.ego_id, random.uniform(10, 25))
            traci.vehicle.setColor(self.ego_id, (255, 0, 0, 255))

            valid_routes = [["start_to_mid", "mid_to_end"], ["end_to_mid", "mid_to_start"]]
            current_edge = traci.vehicle.getRoadID(self.ego_id)
            for route in valid_routes:
                if current_edge == route[0]:
                    traci.vehicle.setRoute(self.ego_id, route)
                    break
        except Exception as e:
            print("Failed to set ego parameters:", e)

        # Adding 15 extra cars ahead of ego
        for i in range(15):
            try:
                traci.vehicle.add(
                    vehID=f"extra_front_{i}",
                    routeID="route0",
                    typeID="car",
                    departPos=str(30 + i * 10),
                    departSpeed=str(random.uniform(5, 12)),
                    departLane="random"
                )
            except:
                continue

        # Adding 15 extra cars behind ego
        for i in range(15):
            try:
                traci.vehicle.add(
                    vehID=f"extra_back_{i}",
                    routeID="route0",
                    typeID="car",
                    departPos=str(max(0, 10 - i * 5)),
                    departSpeed=str(random.uniform(5, 12)),
                    departLane="random"
                )
            except:
                continue

        return self._get_state()

    def step(self, action):
        self._apply_action(action)
        traci.simulationStep()
        self.step_num += 1

        # Adding cut-in vehicles at step 200
        if self.step_num == 200 and not self.cutin_spawned and self.ego_id in traci.vehicle.getIDList():
            try:
                ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
                for i in range(2):
                    cutin_id = f"cutin_{i}"
                    traci.vehicle.add(
                        vehID=cutin_id,
                        routeID="route0",
                        typeID="car",
                        departPos="10",
                        departSpeed="10",
                        departLane=str((ego_lane + 1) % 3)
                    )
                    traci.vehicle.setLaneChangeMode(cutin_id, 512)
                    traci.vehicle.changeLane(cutin_id, ego_lane, 30.0)
                self.cutin_spawned = True
            except Exception as e:
                print("Failed to spawn cut-in cars:", e)

        return self._get_state(), self._compute_reward(), self._is_done(), {}

    def _apply_action(self, action):
        if self.ego_id not in traci.vehicle.getIDList():
            return
        lane_index = traci.vehicle.getLaneIndex(self.ego_id)
        speed = traci.vehicle.getSpeed(self.ego_id)

        if action == 1 and lane_index > 0:
            traci.vehicle.changeLane(self.ego_id, lane_index - 1, 200)
        elif action == 2 and lane_index < 2:
            traci.vehicle.changeLane(self.ego_id, lane_index + 1, 200)
        elif action == 3:
            traci.vehicle.setSpeed(self.ego_id, max(speed - 2, 0))
        elif action == 4:
            traci.vehicle.setSpeed(self.ego_id, min(speed + 2, 30))

    def _get_state(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(3, dtype=np.float32)

        speed = traci.vehicle.getSpeed(self.ego_id)
        lane = traci.vehicle.getLaneIndex(self.ego_id)
        distance = self._get_distance_to_front_vehicle()
        return np.array([speed / 30.0, lane / 2.0, distance / 100.0], dtype=np.float32)

    def _get_distance_to_front_vehicle(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return 100.0
        leader = traci.vehicle.getLeader(self.ego_id, 100)
        return leader[1] if leader else 100.0

    def _compute_reward(self):
        state = self._get_state()
        speed, _, distance = state
        reward = speed * 2

        if distance < 0.1:
            reward -= 5
        if self._has_collision():
            reward -= 10
        return reward

    def _has_collision(self):
        return self.ego_id not in traci.vehicle.getIDList()

    def _is_done(self):
        return self.step_num >= self.max_steps or self._has_collision()

    def close(self):
        if traci.isLoaded():
            traci.close(False)

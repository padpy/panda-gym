from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class ReachCurriculum(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
            )

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.concatenate([np.zeros(2), np.array(self.get_ee_position()), np.zeros(3)])
        return ee_position

    def reset(self) -> None:
        target_position = self._sample_goal()
        # goal vector is [GRASPED_OBJECT(0.0 or 1.0), TOUCHING_OBJECT(0.0 or 1.0), OBJECT_POSITION[float, float, float], PLACE_POSITION[float, float, float]]
        self.goal = np.concatenate([np.zeros(2), target_position, np.zeros(3)])
        self.sim.set_base_pose("target", self.goal[2:5], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal_position = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal_position

    def is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        d = distance(achieved_goal[2:5], desired_goal[2:5])
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(
        self, achieved_goal, desired_goal, info: Dict[str, Any]
    ) -> np.ndarray:
        if achieved_goal.ndim == 2:
            d = distance(achieved_goal[:, 2:5], desired_goal[:, 2:5])
        else:
            d = distance(achieved_goal[2:5], desired_goal[2:5])
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

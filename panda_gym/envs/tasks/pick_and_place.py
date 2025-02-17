from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "sparse",
        distance_threshold: float = 0.045,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array(
            [goal_xy_range / 2, goal_xy_range / 2, goal_z_range]
        )
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
            )

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate(
            [object_position, object_rotation, object_velocity, object_angular_velocity]
        )

        return observation

    def get_achieved_goal(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array([self.grasped(), self.touching_object()], dtype=np.float32),
                self.get_ee_position(),
                np.array(self.sim.get_base_position("object")),
            ]
        )

    def reset(self) -> None:
        target_position = self._sample_target()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", target_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose(
            "object", object_position, np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.goal = np.concatenate([np.array([1.0, 1.0]), object_position, target_position])

    def _sample_target(self) -> np.ndarray:
        """Sample a target."""
        target = np.array(
            [0.0, 0.0, self.object_size / 2]
        )  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        target += noise
        return target

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        if achieved_goal.ndim == 1:
            d = distance(achieved_goal[5:8], desired_goal[5:8])
        else:
            d = distance(achieved_goal[:, 5:8], desired_goal[:, 5:8])
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(
        self, achieved_goal, desired_goal, info: Dict[str, Any]
    ) -> np.ndarray:
        if self.reward_type == "sparse":
            if achieved_goal.ndim == 1:
                d = distance(achieved_goal, desired_goal)
            else:
                d = distance(achieved_goal[:, 5:8], desired_goal[:, 5:8])

            reward = -np.array(d > self.distance_threshold, dtype=np.float32)
            return reward

        else:
            if achieved_goal.ndim == 1:
                obj_d = distance(achieved_goal[2:5], desired_goal[2:5])
                target_d = distance(achieved_goal[5:8], desired_goal[5:8])
                distance_reward = self.distance_threshold - obj_d
                target_reward = self.distance_threshold - target_d
                distance_reward[distance_reward > 0] = 0
                target_reward[target_reward > 0] = 0

                contact_reward = achieved_goal[1] - desired_goal[1]
                grasp_reward = achieved_goal[0] - desired_goal[0]

                if grasp_reward == 0:
                    distance_reward = 0

            else:
                obj_d = distance(achieved_goal[:, 2:5], desired_goal[:, 2:5])
                target_d = distance(achieved_goal[:, 5:8], desired_goal[:, 5:8])
                distance_reward = self.distance_threshold - obj_d
                target_reward = self.distance_threshold - target_d
                distance_reward[distance_reward > 0] = 0
                target_reward[target_reward > 0] = 0

                contact_reward = achieved_goal[:, 1] - desired_goal[:, 1]
                grasp_reward = achieved_goal[:, 0] - desired_goal[:, 0]

                distance_reward[grasp_reward == 0] = 0

            return (
                1 / 6 * distance_reward
                + 2 / 6 * contact_reward
                + 2 / 6 * grasp_reward
                + 1 / 6 * target_reward
            )

    def grasped(self) -> bool:
        left_contact = False
        right_contact = False

        normals = self.sim.get_contact_normals("panda", "object")
        if np.any(normals):
            left_contact = np.any(normals[:, 1] > 0.99)
            right_contact = np.any(normals[:, 1] < -0.99)

        return left_contact and right_contact

    def touching_object(self) -> bool:
        normals = self.sim.get_contact_normals("panda", "object")
        return np.any(normals)

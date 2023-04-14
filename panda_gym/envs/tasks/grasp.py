from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class Grasp(Task):
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
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
        if self.reward_type == "sparse":
            achieved_goal = np.array([self.grasped()], dtype=np.float32)
        else:
            achieved_goal = np.concatenate([np.array([self.grasped(), self.touching_object()], dtype=np.float32), self.get_ee_position()])

        return achieved_goal

    def reset(self) -> None:
        object_position = self._sample_object()
        self.goal = self._sample_goal(object_position)
        self.sim.set_base_pose(
            "object", object_position, np.array([0.0, 0.0, 0.0, 1.0])
        )

    def _sample_goal(self, object_position) -> np.ndarray:
        """Sample a goal."""
        if self.reward_type == "sparse":
            return np.array([1.0], dtype=np.float32)
        else:
            goal = np.concatenate([np.array(
                [1.0, 1.0]
            ), object_position])
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        return np.array(self.grasped(), dtype=bool)

    def compute_reward(
        self, achieved_goal, desired_goal, info: Dict[str, Any]
    ) -> np.ndarray:
        if self.reward_type == "sparse":
            return (achieved_goal - desired_goal).mean()
        else:
            d = distance(achieved_goal[2:], desired_goal[2:])
            distance_reward = self.distance_threshold - d
            distance_reward[distance_reward > 0] = 0

            if achieved_goal.ndim == 1:
                d = distance(achieved_goal[2:], desired_goal[2:])
                distance_reward = self.distance_threshold - d
                distance_reward[distance_reward > 0] = 0

                contact_reward = achieved_goal[1] - desired_goal[1]
                grasp_reward = achieved_goal[0] - desired_goal[0]
            else:
                d = distance(achieved_goal[:,2:], desired_goal[:,2:])
                distance_reward = self.distance_threshold - d
                distance_reward[distance_reward > 0] = 0

                contact_reward = achieved_goal[:,1] - desired_goal[:,1]
                grasp_reward = achieved_goal[:,0] - desired_goal[:,0]

            return 0.25*distance_reward + 0.25*contact_reward + 0.5*grasp_reward

    def grasped(self) -> bool:
        left_contact = False
        right_contact = False

        normals = self.sim.get_contact_normals("panda", "object")
        if np.any(normals):
            left_contact = np.any(normals[:,1] > 0.99)
            right_contact = np.any(normals[:,1] < -0.99)

        return left_contact and right_contact

    def touching_object(self) -> bool:
        normals = self.sim.get_contact_normals("panda", "object")
        return np.any(normals)

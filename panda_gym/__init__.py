import os

from gym.envs.registration import register

MAX_EPISODE_STEPS = 100

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        for action_type in ["continuous", "discrete"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            action_suffix = "Discrete" if action_type == "discrete" else ""
            kwargs = {
                "reward_type": reward_type,
                "control_type": control_type,
                "action_type": action_type,
            }

            register(
                id="PandaReach{}{}{}-v3".format(
                    control_suffix, reward_suffix, action_suffix
                ),
                entry_point="panda_gym.envs:PandaReachEnv",
                kwargs=kwargs,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

            register(
                id="PandaPush{}{}{}-v3".format(
                    control_suffix, reward_suffix, action_suffix
                ),
                entry_point="panda_gym.envs:PandaPushEnv",
                kwargs=kwargs,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

            register(
                id="PandaSlide{}{}{}-v3".format(
                    control_suffix, reward_suffix, action_suffix
                ),
                entry_point="panda_gym.envs:PandaSlideEnv",
                kwargs=kwargs,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

            register(
                id="PandaPickAndPlace{}{}{}-v3".format(
                    control_suffix, reward_suffix, action_suffix
                ),
                entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
                kwargs=kwargs,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

            register(
                id="PandaStack{}{}{}-v3".format(
                    control_suffix, reward_suffix, action_suffix
                ),
                entry_point="panda_gym.envs:PandaStackEnv",
                kwargs=kwargs,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

            register(
                id="PandaFlip{}{}{}-v3".format(
                    control_suffix, reward_suffix, action_suffix
                ),
                entry_point="panda_gym.envs:PandaFlipEnv",
                kwargs=kwargs,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

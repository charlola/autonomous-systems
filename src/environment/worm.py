from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

file_name="env_single/UnityEnvironment"

def load_env(no_graphics=False):
    unity_env = UnityEnvironment(file_name, no_graphics=no_graphics)
    env = UnityToGymWrapper(unity_env)
    return env

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import gym

file_name="env_single/UnityEnvironment"

def load_env(no_graphics=False):
    unity_env = UnityEnvironment(file_name, no_graphics=no_graphics)
    env = UnityToGymWrapper(unity_env)

    return env

def create_gym_env():
    # Pendulum-v0
    env = gym.make('Pendulum-v0')
    return env

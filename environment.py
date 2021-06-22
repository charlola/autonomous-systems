from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import gym

file_name="env_single/UnityEnvironment"

def load_env(no_graphics=False):
    unity_env = UnityEnvironment(file_name, no_graphics=no_graphics)
    env = UnityToGymWrapper(unity_env)

    return env

def create_gym_env(name):
    # Pendulum-v0
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make(name)

    # return env.env to avoid setting done in env.step() after 200 steps
    return env

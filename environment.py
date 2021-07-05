from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym
import os 


def load_env(name, no_graphics=False):
    file_name = os.path.abspath(os.path.join(__file__, os.pardir, "environments", name, "UnityEnvironment"))

    if no_graphics:

        engineConfigChannel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(file_name, no_graphics=no_graphics, side_channels=[engineConfigChannel])
        engineConfigChannel.set_configuration_parameters(time_scale=20.)
    else:
        unity_env = UnityEnvironment(file_name, no_graphics=no_graphics)
    
    env = UnityToGymWrapper(unity_env)

    return env

def create_gym_env(name):
    # Pendulum-v0
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make(name)

    # return env.env to avoid setting done in env.step() after 200 steps
    return env

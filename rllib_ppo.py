from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import environment

env = environment.load_env(name="dynamic_worm", no_graphics=True)
tune.run(PPOTrainer, config={"env": env})
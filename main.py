from src.environment import worm
from src.agent.a2c import A2CAgent
from src.ppo.ppo import PPOAgent
from src.ppo.config import hyperparams
from src import arguments
import matplotlib.pyplot as plt
from ray import tune
import logger
from ray.tune.integration.mlflow import mlflow_mixin
import mlflow
from ray.tune.stopper import ExperimentPlateauStopper

def episode(env, agent, nr_episode, hyperparams):
    state = env.reset()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        if args.graphics:
            env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done, nr_episode)
        state = next_state
        reward = reward.item()
        discounted_return += (hyperparams["gamma"]**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return

@mlflow_mixin
def trainable(config):

    # load environment
    # env = worm.load_env(no_graphics=not args.graphics)
    env = worm.create_gym_env()
    config["env"] = env

    #log params
    params = {}
    for key in config.keys():
        if key != "mlflow":
            params[key] = config[key]
    mlflow.log_params(params)

    # create agent
    agent = PPOAgent(config)

    # define
    returns = list()
    running_reward = None
    running_rewards = list()
    try:
        for i in range(args.episodes):
            score = episode(env, agent, i, config)
            mlflow.log_metric(key="score", value=score, step=i)
            if running_reward is None:
                running_reward = score
            running_reward = running_reward * 0.9 + score * 0.1
            mlflow.log_metric(key="running_reward", value=running_reward, step=i)
            running_rewards.append(running_reward)
            returns.append(score)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        agent.save(args.model)
    tune.report(score=score)  # or running_reward?

    # close environment
    env.close()



if __name__ == "__main__":

    args = arguments.collect()


    analysis = tune.run(
        trainable,
        config=hyperparams
    )

    df = analysis.results_df
    print("Best config:{}".format(analysis.get_best_config(metric="score", mode='max')))


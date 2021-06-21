import gym
import signal
import sys

import mlflow
import ray

import arguments
from src.environment import worm
from src.agent.a2c import A2CAgent
import torch as T
from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin

# Read Arguments
args = arguments.collect()
checkpoint_dir = arguments.get_checkpoint_dir(args.domain, checkpoint=args.checkpoint)
model, start_episode, load_model = arguments.get_checkpoint_model(args.domain, dir=checkpoint_dir,
                                                                  checkpoint=args.checkpoint)


class Trainer:
    def __init__(self, env, params, hyperparams, agent='a2c'):
        self.episode_nr = 0
        self.env = env
        if agent == 'a2c':
            self.agent = A2CAgent(hyperparams, params)
        else:
            self.agent = None  # PPO(hyperparams, params, ....)
        self.params = params
        self.hyperparams = hyperparams

    def episode(self, nr_episode):
        self.episode_nr = nr_episode
        state = self.env.reset()
        discounted_return = 0
        loss = 0
        policy_loss = 0
        entropy_loss = 0
        value_loss = 0
        entropy = 0
        done = False
        time_step = 0
        avg_return = 0
        states = []
        while not done:
            if not self.params['no_graphics']:
                self.env.render()
            # 1. Select action according to policy
            action, entropy = self.agent.policy(state)
            # 2. Execute selected action
            next_state, reward, done, _ = self.env.step(action)
            # 3. Integrate new experience into agent
            if self.params['learn']:
                loss, policy_loss, value_loss, entropy_loss = self.agent.update(state, action, reward, next_state, done)
            states.append(state)
            state = next_state
            avg_return = avg_return + (1 / (time_step + 1)) * (reward - avg_return)
            discounted_return += (self.hyperparams["discount_factor"] ** time_step) * reward
            time_step += 1

        loss = loss.item()
        entropy_item = T.flatten(entropy).mean().item()
        policy_loss = policy_loss.item()
        entropy_loss = entropy_loss.item()
        value_loss = value_loss.item()
        string_format = "{:0>3d}: R {:^16.4f} \tE {:^16.4f} \tL {:^16.4f} \tPL {:^16.4f} \tEL {:^16.4f} \tVL {:^16.4f}"
        if self.episode_nr % 50 == 0:
            print(string_format.format(nr_episode, avg_return, entropy_item, loss, policy_loss, entropy_loss, value_loss))

        return discounted_return, avg_return, entropy_item, loss, policy_loss, entropy_loss, value_loss

    def close(self):
        self.env.close()

    def save(self, episode_nr=None):
        if episode_nr is None:
            episode_nr = self.episode_nr
        self.agent.save(arguments.get_save_model(args.domain, episode_nr, dir=checkpoint_dir, is_best=True))


@mlflow_mixin
def trainable(config):
    # load environment
    if args.domain == "wurmi":
        env = worm.load_env(no_graphics=args.no_graphics)
    elif args.domain == 'car':
        env = gym.make('MountainCarContinuous-v0')
    elif args.domain == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
    elif args.domain == 'pendel':
        env = gym.make('Pendulum-v0')
    else:
        env = None
        print("Choose correct Environment! wurmi | car | lunar | pendel")
        exit(-1)

    params = {
        "learn": args.learn,
        "model": model,
        "load_model": load_model,
        "episodes": args.episodes,
        "no_graphics": args.no_graphics,
        "nr_input_features": env.observation_space.shape[0],
        "nr_actions": env.action_space.shape[0],
        "action_min": -2,
        "action_max": 2
    }

    for key in config.keys():
        if key != "mlflow":
            params[key] = config[key]
    mlflow.log_params(config)

    trainer = Trainer(env, params, hyperparams=config)

    best_result = -9999
    init_count = 0  # is used to make sure, best is not overwritten when model is loaded
    # for episode_nr in range(start_episode, start_episode + params["episodes"] + 1):
    for episode_nr in range(params["episodes"]):
        results = trainer.episode(episode_nr)
        # if args.learn:
        # if results > best_result:
        # best_result = results
        # arguments.remove_best(checkpoint_dir)
        # if init_count > 10:
        # trainer.save(episode_nr)
        # if episode_nr != 0 and episode_nr % args.check_step == 0:
        # trainer.save(episode_nr)
        # init_count += 1
        # tune.report(discounted_return=results[0])
        # discounted_return, avg_return, entropy_item, loss, policy_loss, entropy_loss, value_loss
        mlflow.log_metric(key="discounted_return", value=results[0], step=episode_nr)
        mlflow.log_metric(key="avg_return", value=results[1], step=episode_nr)
        mlflow.log_metric(key="entropy_item", value=results[2], step=episode_nr)
        mlflow.log_metric(key="loss", value=results[3], step=episode_nr)
        mlflow.log_metric(key="policy_loss", value=results[4], step=episode_nr)
        mlflow.log_metric(key="entropy_loss", value=results[5], step=episode_nr)
        mlflow.log_metric(key="value_loss", value=results[6], step=episode_nr)

    trainer.close()


if __name__ == "__main__":
    # define hyperparameter
    config_hyperparams = {
        "gamma": tune.grid_search([0.9, 0.95, 0.99]),
        "alpha": tune.grid_search([0.0001, 0.001, 0.01, 0.1]),
        "discount_factor": tune.grid_search([0.9, 0.95, 0.99]),
        "nr_hidden_units": tune.grid_search([64]),
        "entropy_factor": tune.grid_search([0.0001, 0.001, 0.01, 0.1]),
        "advantage": "TD",
        "mlflow": {
            "experiment_name": "a2c",
            "tracking_uri": "http://159.65.120.229:5000"
        }
    }

    analysis = tune.run(
        trainable,
        name="pendulum",
        local_dir="./ml_flow",
        config=config_hyperparams,
        resources_per_trial={'gpu': 1}
    )

    # data_frame = analysis.results_df
    print("Best config:{}".format(analysis.get_best_config(metric="discount_factor", mode='max')))
    ray.shutdown()

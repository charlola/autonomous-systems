import os, sys
from glob import glob as listdir
sys.coinit_flags=2

import matplotlib.pyplot as plt
from ray import tune
import numpy as np
from torch import nn
import json

import environment
import commandline
from ppo import PPO
from a2c import A2C

def episode(env, agent, nr_episode):
    state = env.reset()
    discounted_return = 0
    total_return = 0
    done = False
    t = 0
    while not done:
        env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        print(action)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        reward = (reward if type(reward) == int else reward.item())
        discounted_return += (args.gamma**t)*reward
        total_return += reward
        t += 1
    print(nr_episode, ":", total_return)
    return total_return


def loop(folder, agent, episodes, logger):

    episode = 0
    batch   = 0

    best = None

    while episode < episodes:
        try:
            # Perform rollout to get batches
            states, next_states, actions, log_probs, rewards, dones, sum_rewards, discounted_return = agent.rollout()

            # Perform updated and learn from rollout
            actor_loss, critic_loss, entropy = agent.learn(states, next_states, actions, log_probs, rewards, dones, discounted_return)

            avg_rewards = np.mean(sum_rewards)
            std_rewads  = np.std(sum_rewards)

            # Add the number of rewards from rollout
            episode += len(sum_rewards)
            batch   += 1

            if not args.tuning or batch % 10 == 0:
                pattern = "\tBatch {: >4d} Episode {: >8d} \tRewards {: >12.2f} \tStd {: >6.6f} \tActor Loss {: >12.6f} \tCritic Loss {: >12.2f} \tEntropy {: >12.2f}"
                print(pattern.format(batch, episode, avg_rewards, std_rewads, actor_loss, critic_loss, entropy))
            
            logger["Total Reward"].extend(sum_rewards)
            logger["Average Reward"].append(avg_rewards)
            logger["Std"].append(std_rewads)
            logger["Actor Loss"].append(actor_loss)
            logger["Critic Loss"].append(critic_loss)
            logger["Avg Std"].append(np.std(logger["Average Reward"][-min(len(logger["Average Reward"]), 10):]))
            logger["Entropy"].append(entropy)

            if best is None or avg_rewards > best:
                agent.model.save(os.path.join(folder, "best"), logger)
                best = avg_rewards

            if batch > 0 and batch % args.checkpoints == 0:
                checkpoint = int(batch / args.checkpoints)
                agent.model.save(os.path.join(folder, "checkpoint_%02d" % checkpoint), logger)

            if args.interrupt and batch > 10 and all(std < 20 for std in logger["Avg Std"][-10:]):
                break
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            raise e
    
    agent.model.save(os.path.join(folder, "final"), logger)

def plot(folder, logger, columns=2, use_average=False, start_avg=1, smoothing=0.9):
    
    # calculate number of culumns
    rows = int((columns-1+len(logger))/columns)

    # create subplots
    fig, axs = plt.subplots(rows, columns, figsize=(14, 10), constrained_layout=True)
    
    # create plot for every entry in the logger
    for i, (name, values) in enumerate(logger.items()):
        # calculate the position of the subplot
        xi = i % columns
        yi = int(i / columns)

        # define x range 
        x = list(range(len(values)-start_avg+1))

        # calculate y values with smoothing
        y = [sum(values[:start_avg]) / start_avg]
        for t, r in enumerate(values[start_avg:]):
            if use_average:
                temp = y[-1] + (1/(t+1)) * (r-y[-1])
            else:
                temp = y[-1] * smoothing + r * (1-smoothing)
            y.append(temp)

        # plot values with name as title
        axs[yi, xi].grid()
        axs[yi, xi].plot(x, y)
        if i == 0:
            std = np.array([np.std(y[-min(i, 300):]) for i in range(len(y))])
            axs[yi, xi].fill_between(x, y-std, y+std, alpha = 0.5)
        elif i == 1:
            std = np.array(logger["Avg Std"])
            axs[yi, xi].fill_between(x, y-std, y+std, alpha = 0.5)

        axs[yi, xi].set_title(name)
    # Adding headline with hyperparams to plot
    fig.suptitle('Env: {environment}, Algorithm: {algorithm}\n Gamma: {gamma}, Critic_Lr: {crit_lr}, '
                 'Actor_Lr: {act_lr},\n Activation_func: {activation_function}, Clip: {clip}'.format(
        environment=args.env_name, algorithm=args.algorithm, gamma=args.gamma, crit_lr=args.critic_lr,
        act_lr=args.actor_lr,  activation_function=args.activation, clip=args.clip), fontsize=16)

    plt.savefig(os.path.join(folder, "image.png"))
    
    if args.graphics:
        plt.show()

def get_hyperparameter():
    return {
        "algorithm":        tune.grid_search(["appo"]), # ,"ppo", "a2c"
        "gamma":            tune.grid_search([0.99]), # 0.8, 0.95, 
        "hidden_units":     tune.grid_search([[64, 64]]), # , [64, 128]
        "activation":       tune.grid_search(["ReLU", "Tanh"]),#
        "clip":             tune.grid_search([0.2]),
        # define config/hyperparams for actor critic
        "critic_discount":  tune.grid_search([0.5]),
        # learning rate
        "actor_lr":         tune.grid_search([0.003, 0.005]), #0.001, 
        "critic_lr":        tune.grid_search([0.003, 0.005]),
        # number of times to update the actor-critic
        "ppo_episodes":     tune.grid_search([4]),
        # number of steps to collect for each trajectory
        "batch_size":       tune.grid_search([4800]),
        "max_step":         tune.grid_search([1600]),  # 0.9 to 1

        # config for mlflow logging
        "mlflow": {
            "experiment_name": "ppo",
            "tracking_uri": "http://159.65.120.229:5000"
        }
    }

def trainable(hyperparameter):
    if type(hyperparameter) == dict:
        # hyperparameter tuning
        for key, value in hyperparameter.items():
            setattr(args, key, value)
    else:
        hyperparameter = {name: getattr(args, name) for name in hyperparameter if name != "mlflow"}

    # load environment
    if args.env_name in ["dynamic_worm", "static_worm"]:
        env = environment.load_env(name=args.env_name, no_graphics=(args.mode == "train"))
    else:
        env = environment.create_gym_env(args.env_name)
    
    # collect env information in args
    args.env = env
    args.state_dim = env.observation_space.shape[0]
    args.act_dim   = env.action_space.shape[0]
    args.action_low  = env.action_space.low[0]
    args.action_high = env.action_space.high[0]
    if hasattr(env, "_max_episode_steps"):
        args.max_step = env._max_episode_steps

    # create agent
    if args.algorithm in ["ppo", "appo"]:
        agent = PPO(args)
    elif args.algorithm in ["a2c", "aa2c"]:
        agent = A2C(args)
    else:
        raise NotImplementedError
    
    logger = logger = {name:[] for name in ["Total Reward", "Average Reward", "Std", "Avg Std", "Actor Loss", "Critic Loss", "Entropy"]}

    # define target folder
    main_folder = os.path.abspath(os.path.join(__file__, os.pardir, "target", args.env_name, args.algorithm))
    

    if args.load is not None:
        agent.model.load(os.path.join(main_folder, args.load), logger)

    if args.mode == "train":
        # create new folder for current training 
        folder = os.path.join(main_folder, "run_%03d" % len(listdir(main_folder + "/run_*")))
        os.makedirs(folder, exist_ok=True)

        # store settings
        with open(os.path.join(folder, "settings.json"), "w") as file:
            json.dump(hyperparameter, file, indent=4)
            
            # train agent
        loop(folder, agent, args.episodes, logger)
        
        # plot results
        plot(folder, logger)
    else:
        if args.load is None:
            raise Exception("No model selected! use --load run_<id>/<name> to do this")

        # show 3 examples of best episode
        for i in range(3):
            episode(env, agent, i)

    # close environment
    env.close()


if __name__ == "__main__":

    # collect arguments from command line
    args = commandline.collect_arguments()

    if args.tuning:
        analysis = tune.run(
            trainable,
            config=get_hyperparameter()
        )
    
    else:
        trainable(list(get_hyperparameter().keys()))

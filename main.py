import sys
sys.coinit_flags=2

import matplotlib.pyplot as plt

import environment
import commandline
from ppo import PPO
from a2c import A2C


def episode(env, agent, nr_episode):
    state = env.reset()
    discounted_return = 0
    done = False
    t = 0
    while not done:
        env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += (args.gamma**t)*reward.item()
        t += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return


def loop(agent, episodes):

    episode = 0
    batch   = 0

    logger = {name:[] for name in ["Total Reward", "Average Reward", "Actor Loss", "Critic Loss"]}

    while episode < episodes:
        try:
            sum_rewards, avg_rewards, actor_loss, critic_loss = agent.learn()

            # Add the number of rewards from rollout
            episode += len(sum_rewards)
            batch   += 1

            pattern = "Batch {: >4d} Episode {: >8d} \tRewards {: >12.2f} \tActor Loss {: >12.6f} \tCritic Loss {: >12.2f}"
            print(pattern.format(batch, episode, avg_rewards, actor_loss, critic_loss))
            
            logger["Total Reward"].extend(sum_rewards)
            logger["Average Reward"].append(avg_rewards)
            logger["Actor Loss"].append(actor_loss)
            logger["Critic Loss"].append(critic_loss)

        except KeyboardInterrupt:
            break
        
    return logger 

def plot(logger, columns=2, use_average=False, start_avg=1, smoothing=0.9):
    
    # calculate number of culumns
    rows = int((columns-1+len(logger))/columns)

    # create subplots
    fig, axs = plt.subplots(rows, columns, figsize=(10, 6), constrained_layout=True)
    
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
        axs[yi, xi].plot(x, y)
        axs[yi, xi].set_title(name)

    plt.show()


if __name__ == "__main__":

    # collect arguments from command line
    args = commandline.collect_arguments()
    
    # load environment
    if args.env_name == "worm":
        env = environment.load_env(no_graphics=not args.graphics)
    else:
        env = environment.create_gym_env(args.env_name)
    
    # collect env information in args
    args.env = env
    args.state_dim = env.observation_space.shape[0]
    args.act_dim   = env.action_space.shape[0]

    # create agent
    if args.algorithm == "ppo":
        agent = PPO(args)

    elif args.algorithm == "a2c":
        agent = A2C(args)
    else:
        raise NotImplementedError
    
    # train agent
    logger = loop(agent, args.episodes)
    
    # plot results
    plot(logger)
    

    if args.graphics:
        for i in range(3):
            episode(env, agent, i)


    '''
    hyperparams["env"] = env

    # create agent
    agent = PPOAgent(hyperparams)

    # define 
    returns = list()
    running_reward = None
    running_rewards = list()
    try:
        for i in range(args.episodes):
            score = episode(env, agent, i, hyperparams)
            if running_reward is None:
                running_reward = score
            running_reward = running_reward * 0.9 + score * 0.1
            running_rewards.append(running_reward)
            returns.append(score)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        agent.save(args.model)

    
    x = range(len(running_rewards))
    y = running_rewards

    plt.plot(x,y)
    plt.title("Progress")
    plt.xlabel("episode")
    plt.ylabel("undiscounted return")
    plt.show()
    '''

    # close environment
    env.close()

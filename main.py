import worm
import commandline
from ppo import PPO
import sys
sys.coinit_flags=2
import matplotlib.pyplot as plt

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
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += (hyperparams["gamma"]**time_step)*reward.item()
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return


if __name__ == "__main__":

    # collect arguments from command line
    args = commandline.collect_arguments()
    
    # load environment
    #env = worm.load_env(no_graphics=not args.graphics)
    env = worm.create_gym_env()
    
    # collect env information in args
    args.env = env
    args.state_dim = env.observation_space.shape[0]
    args.act_dim   = env.action_space.shape[0]

    ###############
    # TESTING PAT #
    ###############
    agent = PPO(args)
    rewards = agent.learn(args.episodes)
    
    columns = 2
    use_average = False
    
    fig, axs = plt.subplots(int((columns-1+len(agent.logging))/columns), 2, figsize=(10, 6), constrained_layout=True)
    
    start_avg = 1

    for i, (name, values) in enumerate(agent.logging.items()):
        xi = i % columns
        yi = int(i / columns)

        x = list(range(len(values)-start_avg+1))
        y = [sum(values[:start_avg]) / start_avg]
        
        for t, r in enumerate(values[start_avg:]):
            if use_average:
                temp = y[-1] + (1/(t+1)) * (r-y[-1])
            else:
                temp = y[-1] * .9 + r * .1
            y.append(temp)

        axs[yi, xi].plot(x, y)
        axs[yi, xi].set_title(name)
    plt.show()
    

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

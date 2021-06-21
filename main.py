from src.environment import worm
from src.ppo_pat.ppo import PPO
from src.agent.a2c import A2CAgent
from src.ppo.ppo import PPOAgent
from src.ppo.config import hyperparams
from src import arguments
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

    #args = arguments.collect()

    # load environment
    #env = worm.load_env(no_graphics=not args.graphics)
    env = worm.create_gym_env()

    ###############
    # TESTING PAT #
    ###############
    model = PPO(env)
    rewards = model.learn(400000)
    
    avg_returns = rewards[:1]
    for t, reward in enumerate(rewards[1:], start=1):
        avg_returns.append(avg_returns[-1] + (1/(t+1))*(reward - avg_returns[-1]))

    x = range(len(avg_returns))
    y = avg_returns

    plt.plot(x, y)
    plt.title("Progress")
    plt.xlabel("timestep")
    plt.ylabel("reward")
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

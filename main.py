from src.environment import worm
from src.agent.a2c import A2CAgent
from src.agent.ppo import PPOAgent

def episode(env, agent, nr_episode, hyperparams):
    state = env.reset()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += (hyperparams["discount_factor"]**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return


if __name__ == "__main__":

    # define parameter
    params = {
        "episodes": 100,
        "no_graphics": False
    }

    # define hyperparameter
    hyperparams = {
        "discount_factor": 0.99,
        "epsilon": 0.2,
        "lambda": 0.95,
        "alpha_actor": 0.001,
        "alpha_critic": 0.001,
        "gamma": 0.99,
    }

    # load environment
    env = worm.load_env(no_graphics=params["no_graphics"])

    # create agent
    agent = PPOAgent(hyperparams)

    # define 
    results = [episode(env, agent, i, hyperparams) for i in range(params["episodes"])]

    # close environment
    env.close()

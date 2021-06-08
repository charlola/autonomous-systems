import arguments
from src.environment import worm
from src.agent.a2c import A2CAgent
from torch.utils.tensorboard import SummaryWriter
import gym


def episode(env, agent, nr_episode, hyperparams, writer):
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
        discounted_return += (hyperparams["discount_factor"] ** time_step) * reward
        time_step += 1
    writer.add_scalar('Loss/epoch', discounted_return, nr_episode)

    print(nr_episode, ":", discounted_return)
    return discounted_return


if __name__ == "__main__":
    use_args = False
    # define parameter
    if use_args:
        args = arguments.collect()
        params = {
            "model": args.model,
            "episodes": args.episodes,
            "no_graphics": args.graphics,
        }
    else:
        params = {
            "model": "a2c",
            "episodes": 100,
            "no_graphics": True,
        }

    # load environment
    env = worm.load_env(no_graphics=params["no_graphics"])
    # env = gym.make('CartPole-v1')

    # define hyperparameter
    hyperparams = {
        "gamma": 0.99,
        "alpha": 0.001,
        "discount_factor": 0.99,
        "nr_hidden_units": 64,
        "nr_input_features": env.observation_space.shape[0],
        "nr_actions": env.action_space.shape[0]
    }

    # create TensorBoard Writer
    writer = SummaryWriter()

    # create agent
    agent = A2CAgent(hyperparams)

    # define
    try:
        results = [episode(env, agent, i, hyperparams, writer) for i in range(params["episodes"])]
    finally:
        agent.save(params["model"])
        writer.flush()

    # close environment
    env.close()

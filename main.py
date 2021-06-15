import gym
import signal
import sys
import arguments
from src.environment import worm
from src.agent.a2c import A2CAgent
import torch as T
from torch.utils.tensorboard import SummaryWriter

best_result = -9999
init_count = 0  # is used to make sure, best is not overwritten when model is loaded
episode_cnt = 0
# Read Arguments
domain = arguments.get_domain()
checkpoint_dir = arguments.get_checkpoint_dir(domain)
model, start_episode, load_model = arguments.get_checkpoint_model(dir=checkpoint_dir)
checkpoint_step = arguments.get_check_step()
no_graphics = arguments.get_no_graphics()
episodes = arguments.get_episodes()
learn = arguments.get_learn()


def signal_handler(sig, frame):
    agent.save(arguments.get_save_model(domain, episode_cnt, dir=checkpoint_dir))
    sys.exit(0)


def episode(env, agent, nr_episode, hyperparams, writer):
    state = env.reset()
    discounted_return = 0
    loss = 0
    entropy = 0
    done = False
    time_step = 0
    states = []
    while not done:
        if not params['no_graphics']: env.render()
        # 1. Select action according to policy
        action, entropy = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        if params['learn']: loss = agent.update(state, action, reward, next_state, done)
        states.append(state)
        state = next_state
        discounted_return += (hyperparams["discount_factor"] ** time_step) * reward
        time_step += 1

    loss_item = loss.item()
    entropy_item = T.flatten(entropy).item()
    if writer is not None:
        writer.add_scalar('Discounted Return/epoch', discounted_return, nr_episode)
        writer.add_scalar('Loss/epoch', loss_item, nr_episode)
        writer.add_scalar('Entropy/epoch', entropy_item, nr_episode)
        writer.add_graph(agent.a2c_net, T.tensor(states, device=agent.device, dtype=T.float32))
    # print(nr_episode, ":", discounted_return)
    string_format = "{:0>3d}: R {:^16.10f} \tL {:^16.10f} \tE {:^16.10f}"
    print(string_format.format(nr_episode, discounted_return, loss_item, entropy_item))
    # print(nr_episode, ":", "R", discounted_return, "\tL", loss_item, "\tE", entropy_item)

    return discounted_return


if __name__ == "__main__":
    # load environment
    if domain == "wurmi":
        env = worm.load_env(no_graphics=no_graphics)
    elif domain == 'car':
        env = gym.make('MountainCarContinuous-v0')
    elif domain == 'lunar':
        env = gym.make('LunarLander-v2')
    elif domain == 'pendel':
        env = gym.make('Pendulum-v0')
    else:
        print("Choose correct Environment! wurmi | car | lunar | pendel")
        exit(-1)

    params = {
        "learn": learn,
        "model": model,
        "load_model": load_model,
        "episodes": episodes,
        "no_graphics": no_graphics,
        "nr_input_features": env.observation_space.shape[0],
        "nr_actions": env.action_space.shape[0],
    }

    # define hyperparameter
    hyperparams = {
        "gamma": 0.99,
        "alpha": 0.001,
        "discount_factor": 0.99,
        "nr_hidden_units": 64,
        "entropy_factor": 0.01,
        "advantage": "RL",
    }

    # create TensorBoard Writer
    # writer = SummaryWriter()
    writer = None

    signal.signal(signal.SIGINT, signal_handler)
    # create agent
    agent = A2CAgent(hyperparams, params)

    # define
    try:
        for episode_cnt in range(start_episode, params["episodes"] + 1):
            results = episode(env, agent, episode_cnt, hyperparams, writer)
            if learn:
                if results > best_result:
                    best_result = results
                    arguments.remove_best(checkpoint_dir)
                    if init_count > 10:
                        agent.save(arguments.get_save_model(domain, episode_cnt, dir=checkpoint_dir, is_best=True))
                if episode_cnt != 0 and episode_cnt % checkpoint_step == 0:
                    agent.save(arguments.get_save_model(domain, episode_cnt, dir=checkpoint_dir))
            init_count += 1
    finally:
        if learn:
            agent.save(arguments.get_save_model(domain, episode_cnt, dir=checkpoint_dir))
        if writer is not None:
            writer.flush()

    # close environment
    env.close()

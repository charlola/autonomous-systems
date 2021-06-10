import gym
import signal
import sys
import os
import argparse
import re
from datetime import datetime
from src.environment import worm
from src.agent.a2c import A2CAgent
from torch.utils.tensorboard import SummaryWriter

learn = True

best_result = -9999
domain = "wurmi"

checkpoint_path = ""
checkpoint_dir = ""
load_model = False
start = 0
model = ""

checkpoint_step = 500
no_graphics = True


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


parser = argparse.ArgumentParser(description='Add some Arguments to the program.')
parser.add_argument("--domain", help="Enter 'wurmi' or 'car'. Wurmi is default", type=str)
parser.add_argument("--checkpoint",
                    help="Enter the path to the checkpoint you want to start from. Make sure it fits your Environment!",
                    type=str)
parser.add_argument("--learn", help="Enter True for Training mode", type=bool)
parser.add_argument("--check_step", help="The amount of Episodes when a checkpoint will be created", type=int)
parser.add_argument("--no_graphics", help="True or false", type=bool)
args = parser.parse_args()

if args.domain:
    domain = args.domain

if args.checkpoint:
    checkpoint_path = args.checkpoint
    if os.path.isfile(checkpoint_path):
        if checkpoint_path.endswith("_best.nn"):
            start = get_trailing_number(checkpoint_path.replace("_best.nn", ""))
        else:
            start = get_trailing_number(checkpoint_path.replace(".nn", ""))
        print("Start: ", start)
        load_model = True
        checkpoint_dir = os.path.dirname(checkpoint_path) + "\\"
        model = checkpoint_path
    else:
        print("Error reading Checkpoint-File '", checkpoint_path, "'")
        exit(-1)
else:
    # Create new checkpoint file
    cur_time = datetime.now()
    checkpoint_dir = "models\\" + domain + "_" + str(cur_time.day) + str(cur_time.month) + str(cur_time.year)[:2] + \
                     "-" + str(cur_time.hour) + "_" + str(cur_time.minute) + "\\"
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    model = checkpoint_dir + domain + ".nn"

if args.check_step:
    checkpoint_step = args.check_step

if args.learn:
    learn = args.learn

if args.no_graphics:
    no_graphics = args.no_graphics


def remove_best():
    test = os.listdir(checkpoint_dir)
    for item in test:
        if item.endswith("_best.nn"):
            os.remove(os.path.join(checkpoint_dir, item))


def signal_handler(sig, frame):
    agent.save(params["model"])
    sys.exit(0)


def episode(env, agent, nr_episode, hyperparams, writer):
    state = env.reset()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        if not params['no_graphics']: env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        if params['learn']: agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += (hyperparams["discount_factor"] ** time_step) * reward
        time_step += 1
    writer.add_scalar('Loss/epoch', discounted_return, nr_episode)
    print(nr_episode, ":", discounted_return)
    return discounted_return


if __name__ == "__main__":
    # load environment
    if domain is "wurmi":
        env = worm.load_env(no_graphics=no_graphics)
    else:
        env = gym.make('MountainCarContinuous-v0')

    params = {
        "learn": learn,
        "model": model,
        "load_model": load_model,
        "episodes": 10000,
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
        "advantage": "TD",
    }

    # create TensorBoard Writer
    writer = SummaryWriter()
    signal.signal(signal.SIGINT, signal_handler)
    # create agent
    agent = A2CAgent(hyperparams, params)

    # define
    try:
        for i in range(start, params["episodes"]):
            results = episode(env, agent, i, hyperparams, writer)
            if learn:
                if results > best_result:
                    best_result = results
                    remove_best()
                    agent.save(checkpoint_dir + domain + "-" + str(i) + "_best.nn")
                if i != 0 and i % checkpoint_step == 0:
                    agent.save(checkpoint_dir + domain + "_" + str(i) + ".nn")
                last_step = i
    finally:
        if learn:
            agent.save(checkpoint_dir + domain + "_" + str(i) + ".nn")
        writer.flush()

    # close environment
    env.close()

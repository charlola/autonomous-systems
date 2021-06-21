import argparse
import re
import os
from datetime import datetime


def str2bool(str):
    if str.lower() in ["false", "f", "0"]:
        return False
    return True


def collect():
    parser = argparse.ArgumentParser(description='Setup your Environment.')
    parser.add_argument("-d", "--domain", help="Enter 'wurmi', 'car', 'pendel' or 'lunar'. Wurmi is default",
                        default="wurmi", type=str)
    parser.add_argument("-c", "--checkpoint",
                        help="Enter the path to the checkpoint you want to start from. Make sure it fits your Environment!",
                        type=str)
    parser.add_argument("-l", "--learn", help="Enter True for Training mode", default=True, type=str2bool)
    parser.add_argument("-cs", "--check_step", help="The amount of Episodes when a checkpoint will be created",
                        default=500, type=int)
    parser.add_argument("-g", "--no_graphics", help="True or false", default=True, type=str2bool)
    parser.add_argument("-e", "--episodes", help="Set amnt. of Episodes", default=10000, type=int)
    parser.add_argument("-f", '--file', help="Enter path to a file containing arguments.(--file args.txt)",
                        default="args.txt", type=open,
                        action=LoadFromFile)
    return parser.parse_args()


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def get_checkpoint_dir(domain, dir="", checkpoint=""):
    if checkpoint:
        return os.path.dirname(checkpoint) + "\\"
    elif not dir:
        cur_time = datetime.now()
        checkpoint_dir = "models\\" + domain + "_" + str(cur_time.day) + str(cur_time.month) + str(cur_time.year)[:2] + \
                         "-" + str(cur_time.hour) + "_" + str(cur_time.minute) + "\\"
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return checkpoint_dir
    else:
        return dir


def get_checkpoint_model(domain, dir="", best=False, checkpoint=""):
    """
    :return: Path to checkpoint file, Epoch where to start, Should load the model?
    """
    if checkpoint:
        if os.path.isfile(checkpoint):
            if checkpoint.endswith("_best.nn"):
                start = get_trailing_number(checkpoint.replace("_best.nn", ""))
            else:
                start = get_trailing_number(checkpoint.replace(".nn", ""))
            checkpoint_dir = os.path.dirname(checkpoint) + "\\"
            return checkpoint_dir, start, True
        else:
            print("Error reading Checkpoint-File '", checkpoint, "'")
            exit(-1)
    elif dir:
        checkpoint_dir = dir
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if best:
            return checkpoint_dir + domain + "_best.nn", 0, False
        else:
            return checkpoint_dir + domain + ".nn", 0, False

    else:
        # Create new checkpoint file
        checkpoint_dir = get_checkpoint_dir(domain)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return checkpoint_dir + domain + ".nn", 0, False


def remove_best(checkpoint_dir):
    test = os.listdir(checkpoint_dir)
    for item in test:
        if item.endswith("_best.nn"):
            os.remove(os.path.join(checkpoint_dir, item))


def get_save_model(domain, last_episode, dir="", is_best=False):
    dir = get_checkpoint_dir(domain, dir)
    return dir + domain + "-" + str(last_episode) + ("_best.nn" if is_best else ".nn")


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)

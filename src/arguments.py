import argparse

def str2bool(str):
    if str.lower() in ["false", "f", "0"]:
        return False
    return True

def collect():
    parser = argparse.ArgumentParser(description='Define the parameter for Captain Wurmi')
    parser.add_argument("-e", "--episodes", default=15, metavar='N', type=int,
                        help='Define the number of episodes')

    parser.add_argument("-m", "--model", default="ppo.nn", metavar='S', type=str, help='Define the model to be used/overwritten')
    parser.add_argument("-g", "--graphics", default=False, metavar='B', type=str2bool, help="Define if graphics should be shown")

    return parser.parse_args()
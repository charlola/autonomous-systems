import argparse

def collect_arguments():
    def str2bool(str):
        if str.lower() in ["false", "f", "0", False]:
            return False
        return True

    parser = argparse.ArgumentParser(
        description='Define the parameter for Captain Wurmi')
    
    # helper
    parser.add_argument("-e", "--episodes",      default=100,           type=int,      help="Define the number of episodes")
    parser.add_argument("-g", "--graphics",      default=False,         type=str2bool, help="Define if graphics should be shown")
    parser.add_argument("--env_name",            default="Pendulum-v0", type=str,      help="Define the environment")
    parser.add_argument("--use_hyperparameter", default=False,         type=str2bool, help="Define the environment")
    
    # net
    parser.add_argument("--hidden_units", default=64, metavar='I', type=int, help="Number of hidden units")

    # hyperparameter
    parser.add_argument("--gamma",            default=0.95,   type=float, help="Gamma")
    parser.add_argument("--clip",             default=0.2,    type=float, help="Clipping Value")
    parser.add_argument("--actor_lr",         default=0.005,  type=float, help="Learning Rate")
    parser.add_argument("--critic_lr",        default=0.005,  type=float, help="Learning Rate")
    parser.add_argument("--k",                default=5,      type=int,   help="n_updates_per_iteration")
    parser.add_argument("-a", "--algorithm",  default="ppo",  type=str,   help="the algorithm ppo | a2c")
    parser.add_argument("--batch_size",       default=4800,   type=int,   help="timesteps_per_batch")
    parser.add_argument("--max_step",         default=1600,   type=int,   help="max_timesteps_per_episode")

    ############################################################################

    parser.add_argument("--noise",          default=0.001,       type=float, help="Noise Factor")
    parser.add_argument("--value",          default=0.5,         type=float, help="Value Factor")
    parser.add_argument("--mode",           default="train",     type=str,   help='Mode to evaluate (train|test)')
    parser.add_argument("--model",          default="ppo.nn",    type=str,   help='Define the model to be used/overwritten')
    parser.add_argument("--ppo_episodes",   default="4",         type=int,   help="Number of PPO Episodes")
    parser.add_argument("--advantage",      default="ADVANTAGE", type=str,   help="Choose the advantage function (REINFORCE | TEMPORAL | ADVANTAGE)")
    parser.add_argument("--max_grad_norm",  default=0.5,         type=int,   help="Maximum of gradient")
    
    return parser.parse_args()
import argparse

def collect_arguments():
    def str2bool(s):
        if s == False or s.lower() in ["false", "f", "0"]:
            return False
        return True

    def str2list(s):
        return list(map(int, s.split()))
    
    parser = argparse.ArgumentParser(
        description='Define the parameter for Captain Wurmi')
    
    # program arguments
    parser.add_argument("-e", "--episodes",     default=30000,          type=int,       help="Define the number of episodes")
    parser.add_argument("-g", "--graphics",     default=False,          type=str2bool,  help="Define if graphics should be shown")
    parser.add_argument("-a", "--algorithm",    default="appo",         type=str,       help="Define the algorithm ppo | appo |a2c")
    parser.add_argument("-n", "--env_name",     default="static_worm",  type=str,       help="Define the environment")
    parser.add_argument("-t", "--tuning",       default=False,          type=str2bool,  help="Define if hyperparameter tuning should be used")
    parser.add_argument("-i", "--interrupt",    default=False,          type=str2bool,  help="Define if the algorithm can be interrupted due to low std")
    parser.add_argument("-c", "--checkpoints",  default=20,             type=int,       help="The intervall of batches to store checkpoints of the net")
    parser.add_argument("-l", "--load",         default=None,           type=str,       help="Load the weights for the net from")
    parser.add_argument("-m", "--mode",         default="train",        type=str,       help='Mode to evaluate (train|test)')

    # net
    parser.add_argument("--hidden_units",       default="64 64",        type=str2list,  help="Hidden units as list separated by single space e.g. '64 64'")
    parser.add_argument("--activation",         default="Tanh",         type=str,       help="Define if hyperparamete training should be used (ReLU | Tanh | ...)")

    # hyperparameter
    parser.add_argument("--gamma",              default=0.995,          type=float,     help="Gamma")
    parser.add_argument("--gae_lambda",         default=0,              type=float,     help="Lambda")
    parser.add_argument("--clip",               default=0.2,            type=float,     help="Clipping Value")
    parser.add_argument("--actor_lr",           default=0.005,          type=float,     help="Learning Rate")
    parser.add_argument("--critic_lr",          default=0.005,          type=float,     help="Learning Rate")
    parser.add_argument("--ppo_episodes",       default=5,              type=int,       help="n_updates_per_iteration")
    parser.add_argument("--batch_size",         default=5000,           type=int,       help="timesteps_per_batch")
    parser.add_argument("--mini_batch_size",    default=32,             type=int,       help="mini batches each ppo episode")
    parser.add_argument("--shuffle_mini_batch", default=True,           type=str2bool,  help="shuffle mini batch")
    parser.add_argument("--max_step",           default=1000,           type=int,       help="max_timesteps_per_episode")
    parser.add_argument("--noise",              default=0,              type=float,     help="Noise Factor") # 0.001
    parser.add_argument("--noise_decay",        default="linear",       type=str,       help="Noise Decay Type (linear | geometric)")
    parser.add_argument("--normalize",          default="advantage",    type=str,       help="Define what should normalized")
    parser.add_argument("--advantage",          default="advantage",    type=str,       help="Choose the advantage function (reinforce | temporal | advantage)")
    parser.add_argument("--max_grad_norm",      default=0,              type=float,     help="Maximum of gradient")
    
    return parser.parse_args()
# define parameter
params = {
    "episodes": 100,
    "no_graphics": False,
    "state_dim": 57,
    "action_dim": 9,
}

hyperparams = {

    # define hyperparameter
    "discount_factor": 0.99,
    "epsilon": 0.2,
    "lambda": 0.95,
    "alpha_actor": 0.001,
    "alpha_critic": 0.001,
    "gamma": 0.99,
    
    # define config/hyperparams for actor critic
    "nr_hidden_units": 64,
    "action_std_init": 0.6,

    # define config/hyperparams for PPO agent
    "epsilon": 0,
    "k": 0,
    "step_size": 10
}

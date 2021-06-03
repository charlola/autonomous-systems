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

# define config/hyperparams for actor critic
ac_config = {
    "nr_input_features": 0,
    "action_dim": 0,
    "nr_hidden_units": 64,
    "device": "cpu",
    "action_std_init": 0.6
}

# define config/hyperparams for PPO agent
ppoa_config = {
    "alpha_actor": 0,
    "alpha_critic": 0,
    "gamma": 0,
    "device": "cpu",
    "epsilon": 0,
    "k": 0
}




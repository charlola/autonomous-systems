hyperparams = {

    # define parameter for neuronal net
    "nr_hidden_units": 64,

    # discount rate
    "gamma": 0.99,         # 0.99 (most common), 0.8 to 0.9997
    
    # learning rate
    "alpha": 5e-6,        # 0.003 to 5e-6

    # define config/hyperparams for actor critic
    "critic_discount": 0.5, # 0.5, 1
    "noise_factor": 0.01,   # 0 to 0.01

    # clipping parameter 
    "epsilon": 0.1,         # 0.1, 0.2, 0.3

    # number of times to update the actor-critic
    "k": 3,

    # number of steps to collect for each trajectory
    "step_size": 128,

    # define config/hyperparams for PPO agent
    "batch_size": 8,
    
    # define hyperparameter
    "lambda": 0.95,         # 0.9 to 1

}

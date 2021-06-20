hyperparams = {

    # define parameter for neuronal net
    "nr_hidden_units": 64,

    # discount rate
    "gamma": 0.99,         # 0.99 (most common), 0.8 to 0.9997
    
    # learning rate
    "alpha": 0.001,        # 0.003 to 5e-6

    # factor for critic loss
    "critic_discount": 0.5, # 0.5, 1

    # factor for noise in loss
    "noise_factor": 0.01,   # 0 to 0.01

    # clipping parameter 
    "epsilon": 0.2,         # 0.1, 0.2, 0.3

    # number of times to update the actor-critic
    "k": 4,

    # number of steps to collect for each trajectory
    "step_size": 32,

    # batch size within ppo episodes
    "batch_size": 12,
}

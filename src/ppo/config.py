from ray import tune

hyperparams = {

    # define parameter for neuronal net
    "nr_hidden_units": 64,

    # discount rate
    "gamma": 0.99,  # 0.99 (most common), 0.8 to 0.9997

    # learning rate
    "alpha":0.001,  # 0.003 to 5e-6

    # define config/hyperparams for actor critic
    "critic_discount": 0.5,
    "noise_factor": 0.01,  # 0 to 0.01
    "noise_decay": 0.95,
    "noise_min": 0.01,

    # clipping parameter
    "epsilon": 0.1,  # 0.1, 0.2, 0.3
    "epsilon_decay": 0.995,
    "epsilon_min": 0.001,

    # number of times to update the actor-critic
    "k": 4,

    # number of steps to collect for each trajectory
    "step_size": 32,

    # define config/hyperparams for PPO agent
    "batch_size": 32,

    # define hyperparameter
    "lambda": 0.95,  # 0.9 to 1

    # config for mlflow logging
    "mlflow": {
        "experiment_name": "ppo",
        "tracking_uri": "http://159.65.120.229:5000"
    }
}

ray_hyperparams = {

    # define parameter for neuronal net
    "nr_hidden_units": tune.grid_search([64]),

    # discount rate
    "gamma": tune.grid_search([0.99, 0.95, 0.8]),         # 0.99 (most common), 0.8 to 0.9997
    
    # learning rate
    "alpha": tune.grid_search([0.00001, 0.0001, 0.001]),        # 0.003 to 5e-6

    # define config/hyperparams for actor critic
    "critic_discount": tune.grid_search([0.5, 1]),
    "noise_factor": tune.grid_search([0,0.005, 0.01]),   # 0 to 0.01
    "noise_decay": 0.95,
    "noise_min": 0.01,

    # clipping parameter 
    "epsilon": 0.1,         # 0.1, 0.2, 0.3
    "epsilon_decay": 0.995,
    "epsilon_min": 0.001,

    # number of times to update the actor-critic
    "k": 4,

    # number of steps to collect for each trajectory
    "step_size": 128,

    # define config/hyperparams for PPO agent
    "batch_size": 32,
    
    # define hyperparameter
    "lambda": 0.95,     # 0.9 to 1

    #config for mlflow logging
    "mlflow": {
            "experiment_name": "ppo",
            "tracking_uri": "http://159.65.120.229:5000"
        }

}

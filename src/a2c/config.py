from ray import tune

# define hyperparameter
hyperparams = {
    "gamma": 0.99,
    "alpha": 0.001,
    "discount_factor": 0.99,
    "nr_hidden_units": 64,
    "entropy_factor": 0.001,
    "advantage": "RL",
}

hyperparams = {

    # define parameter for neuronal net
    "nr_hidden_units": 64,

    # discount rate
    "gamma": tune.grid_search([0.99, 0.95]),  # 0.99 (most common), 0.8 to 0.9997

    # learning rate
    "alpha": 0.0001,  # 0.003 to 5e-6

    # define config/hyperparams for actor critic
    "critic_discount": 0.5,  # 0.5, 1
    "noise_factor": 0.5,  # 0 to 0.01
    "noise_decay": 0.95,
    "noise_min": 0.01,

    # clipping parameter
    "epsilon": 0.1,  # 0.1, 0.2, 0.3
    "epsilon_decay": 0.995,
    "epsilon_min": 0.001,

    # number of times to update the actor-critic
    "k": 4,

    # number of steps to collect for each trajectory
    "step_size": 128,

    # define config/hyperparams for PPO agent
    "batch_size": 32,

    # define hyperparameter
    "lambda": 0.95,  # 0.9 to 1

    # config for mlflow logging
    "mlflow": {
        "experiment_name": "a2c",
        "tracking_uri": "http://159.65.120.229:5000"
    }


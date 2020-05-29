args = {
    # Arguments for input data for agent, and auto-encoders
    "input": {
        "batch_size": 128,
        "num_channels": 1,
        "input_size": (64, 64),
    },

    # Arguments for the auto-encoders
    "BetaVAE": {
        "Z_dim": 64,
        "C_max": 25,
        "C_stop_iter": 1e5,
        "lr": 1e-4,
    },



    # Arguments for the reinforcement learning
    "RL": {
        "gamma": 0.99,
        "lr": 2.5e-4,
    },

}

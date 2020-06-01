args = {
    # Arguments for input data for agent, and auto-encoders
    "hyper_parameters": {
        "batch_size": 1,
        "num_channels": 1,
        "input_size": (128, 128),
        "z_dim": 64,
        "dataset": '/home/samir/Desktop/repositories/auto-features-extraction-for-RL/src/'
    },

    # Arguments for the auto-encoders
    "AAE": {
        "encoder_lr": 0.0006,
        "decoder_lr": 0.0006,
        "discriminator_lr": 0.0008,
    },
    "BetaVAE": {
        "num_epochs": 1e6,
        "C_max": 25,
        "C_stop_iter": 1e5,
        "Gamma": 120,
        "lr": 1e-4,
    },

    "WassersteinAE": {
        "reg_weight": 100,
        "lr": 1e-4,
    },
    "ModifiedVAE": {
        "Z_dim": 10,
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

args = {
    # Arguments for input data for agent, and auto-encoders
    "hyper_parameters": {
        "num_epochs": 1.5e3,
        "batch_size": 1024,
        "num_channels": 3,
        "input_size": (128, 128),
        "z_dim": 64,
        "dataset": '/home/samir/Desktop/repositories/auto-features-extraction-for-RL/src/'

    },

    # Arguments for the auto-encoders
    "aae": {
        "encoder_lr": 0.0006,
        "decoder_lr": 0.0006,
        "discriminator_lr": 0.0008,
    },
    "beta_vae": {
        "C_max": 100,
        "C_stop_iter": 1e4,
        "Gamma": 1,
    },

    "wasserstein_ae": {
        "reg_weight": 100,
    },

    "vanilla_vae": {
        "Z_dim": 10,
        "M_N": 0.005,
    },

    # Arguments for the reinforcement learning
    "RL": {
        "gamma": 0.99,
        "lr": 2.5e-4,
    },

}

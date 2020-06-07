import os
import torch

from src.reinforcement_learning.scenario.rl_train_scenario import RLTrainScenario
#from reinforcement_learning.ddpg.ddpg_agent import DDPGAgent
from src.reinforcement_learning.sac.sac_agent import SACAgent
from src.reinforcement_learning.env.wrappers.donkey_wrapper import DonkeyCarEnvironment
from src.reinforcement_learning.scenario.progress.neptune_progress_reporter import NeptuneProgressReporter
from src.reinforcement_learning.scenario.progress.console_progress_reporter import ConsoleProgressReporter

# from features_extraction.JigsawVAE.model import JigsawVAE
# from features_extraction.CLR.checkCLRonRL import CustomSimCLR50, CustomSimCLR18
from src.features_extraction.vanilla_vae.model import VanillaVAE
from src.features_extraction.beta_vae.model import BetaVAE
from src.features_extraction.wasserstein_ae.model import WassersteinAE
from src.features_extraction.aae.model import AdversarialAE

from framework.CustomModels import CustomSimCLR18

from src.args import args
from src.utils import get_fixed_hyper_param


def rescale_state(state, minimum=-3500.0, maximum=3000.0):
    length = maximum - minimum
    return (state - minimum) / length


def main():

    # agent = DDPGAgent((128, 128),
    #                   env.get_action_space().shape[0],
    #                   env.get_action_space().low,
    #                   env.get_action_space().high,
    #                   gamma=0.99,
    #                   tau=1e-2,
    #                   buffer_size=100000,
    #                   critic_lr=1e-4,
    #                   actor_lr=1e-4,
    #                   force_cpu=True)

    # autoencoder = JigsawVAE(image_size=128, channel_num=3, kernel_num=224, z_size=64).cpu()
    # autoencoder.load_state_dict(torch.load('best_model.pth', map_location=lambda storage, loc: storage))
    # autoencoder.cpu()

    # clr_model = CustomSimCLR18(5, 64, 128)
    # checkpoint = torch.load("CustomSimCLR18best_model.pth", map_location=lambda storage, loc: storage)
    # clr_model.load_state_dict(checkpoint['state_dict'])

    _, num_of_channels, input_size, z_dim = get_fixed_hyper_param(args['hyper_parameters'])

    # vae = VanillaVAE(z_dim, num_of_channels, input_size)
    # vae.load_state_dict(torch.load("../saved_model/VanillaVAE.best.pth"))
    # vae = BetaVAE(z_dim, num_of_channels, input_size).cpu()
    # vae.load_state_dict(torch.load("../saved_model/BetaVAE.best.pth", map_location=lambda storage, loc: storage))
    # vae = WassersteinAE(z_dim, num_of_channels, input_size).cpu()
    # vae.load_state_dict(torch.load("../saved_model/WassersteinAE.best.pth", map_location=lambda storage, loc: storage))
    # vae = AdversarialAE(z_dim, num_of_channels, input_size).cpu()
    # vae.load_state_dict(torch.load("../saved_model/AdversarialAE.best.pth", map_location=lambda storage, loc: storage))
    # vae = vae.cpu()
    # vae.eval()
    # encode_fn = lambda img: rescale_state(vae.encoder.forward(torch.FloatTensor(img).unsqueeze(dim=0)).squeeze().numpy(),
    #                                       )minimum = -400000, maximum = 300000
    model = CustomSimCLR18(64, 256, pretrained=True).cpu()
    model.load_state_dict(torch.load("../saved_model/SimCLR18.best.pth", map_location=lambda storage, loc: storage)["state_dict"])
    model = model.cpu()
    model.eval()
    encode_fn = lambda img: model.forward(torch.FloatTensor(img).unsqueeze(dim=0))[0].squeeze().numpy()

    env = DonkeyCarEnvironment(os.path.abspath("./reinforcement_learning/env/third_party_environments/"),
                               encoder_fn=encode_fn)

    agent = SACAgent(env.observation_space.shape[0],
                     env.action_space.shape[0],
                     [env.action_space.low, env.action_space.high],
                     gamma=0.99,
                     tau=0.01,
                     alpha=0.2,
                     critic_lr=3e-4,
                     actor_lr=3e-4,
                     alpha_lr=3e-4,
                     buffer_size=30000,
                     force_cpu=True)

    # agent = SACAgent(env.observation_space, env.action_space,
    #                  gamma=0.99,
    #                  alpha=0.2,
    #                  polyak=0.01,
    #                  buffer_size=30000,
    #                  lr=3e-4,
    #                  force_cpu=True)

    # agent = DDQNAgent(env.env_.observation_space.shape,
    #                       env.action_space.shape[0],
    #                       [env.action_space.low, env.action_space.high],
    #                       gamma=0.99,
    #                       tau=1e-2,
    #                       buffer_size=100000,
    #                       lr=1e-3)

    scenario = RLTrainScenario(env, agent,
                               progress_reporter=NeptuneProgressReporter("timeescaper/rl-full-image", "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMmE1NDY1MjQtOTJhZS00NjJmLTlkMmYtNTIyOWJlNjEyMWJhIn0="),
                               n_episodes=10000,
                               max_steps=5000,
                               batch_size=64)
    scenario.run()


if __name__ == "__main__":
    main()

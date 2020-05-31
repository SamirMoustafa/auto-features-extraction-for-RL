from abc import ABC


class RLAgent(ABC):
    def get_action(self, state):
        pass

    def train_step(self, batch_size):
        pass

    def set_eval_mode(self, mode):
        pass

    def save_checkpoint(self, checkpoint_path):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass

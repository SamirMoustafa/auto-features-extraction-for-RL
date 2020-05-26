from abc import abstractmethod, ABCMeta

from torch import nn


class Encoder(metaclass=ABCMeta):
    def __init__(self):
        super(Encoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass


class Decoder(metaclass=ABCMeta):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass


class Bottleneck(metaclass=ABCMeta):
    def __init__(self):
        super(Bottleneck, self).__init__()
        pass

    @abstractmethod
    def forward(self, **kwargs):
        pass


class LossFunction(metaclass=ABCMeta):
    def __init__(self):
        super(LossFunction, self).__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
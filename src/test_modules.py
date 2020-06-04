import unittest

from src.utils import game_data_loaders


class TestModelMethods(unittest.TestCase):

    def test_model(self, model, loss):
        data_loaders = game_data_loaders()
        train_loaders, val_loaders = data_loaders['train'], data_loaders['val']

        print('num_batches_train:', len(train_loaders))
        print('num_batches_val:', len(val_loaders))
        print('x_batch_shape:', next(iter(train_loaders))[0].shape)
        print('y_batch_shape:', next(iter(train_loaders))[1].shape)

        # input
        x = next(iter(train_loaders))

        # test model
        out_puts = model(x)
        loss(x, *out_puts)
        self.assertEqual(True, True, msg='Equal')

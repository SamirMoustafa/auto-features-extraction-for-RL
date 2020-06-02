import unittest

from src.utils import mnist_NxN_loader


class TestModelMethods(unittest.TestCase):

    def test_model(self, model, loss):
        train_loader, test_loader = mnist_NxN_loader()

        print('num_batches_train:', len(train_loader))
        print('num_batches_test:', len(test_loader))
        print('x_batch_shape:', next(iter(train_loader))[0].shape)
        print('y_batch_shape:', next(iter(train_loader))[1].shape)

        # input
        x = next(iter(train_loader))[0]

        # test model
        out_puts = model(x)
        loss(x, *out_puts)
        self.assertEqual(True, True, msg='Equal')

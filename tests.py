import unittest
import numpy as np
from numpy.random import randint, random_sample
from scipy import stats
import torch
from torch.autograd import Variable

import nninit


class TestNNInit(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        torch.manual_seed(123)

    def _is_normal(self, tensor, mean, std):
        if isinstance(tensor, Variable):
            tensor = tensor.data
        p_value = stats.kstest(tensor.numpy().flatten(), 'norm', args=(mean, std)).pvalue
        return p_value > 0.0001

    def _is_uniform(self, tensor, a, b):
        if isinstance(tensor, Variable):
            tensor = tensor.data
        p_value = stats.kstest(tensor.numpy().flatten(), 'uniform', args=(a, (b - a))).pvalue
        return p_value > 0.0001

    def _create_random_nd_tensor(self, dims, size_min, size_max, as_variable):
        size = [randint(size_min, size_max + 1) for _ in range(dims)]
        tensor = torch.zeros(size)
        if as_variable:
            tensor = Variable(tensor)
        return tensor

    def _random_float(self, a, b):
        return (b - a) * random_sample() + a

    def test_uniform(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)
                a = self._random_float(-3, 3)
                b = a + self._random_float(1, 5)
                nninit.uniform(input_tensor, a=a, b=b)
                assert self._is_uniform(input_tensor, a, b)

    def test_normal(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)
                mean = self._random_float(-3, 3)
                std = self._random_float(1, 5)
                nninit.normal(input_tensor, mean=mean, std=std)

                assert self._is_normal(input_tensor, mean, std)

    def test_constant(self):
        for as_variable in [True, False]:
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5, as_variable=as_variable)
                val = self._random_float(1, 10)
                nninit.constant(input_tensor, val)
                if as_variable:
                    input_tensor = input_tensor.data

                assert np.allclose(input_tensor.numpy(), input_tensor.clone().fill_(val).numpy())

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                with self.assertRaises(ValueError):
                    nninit.xavier_uniform(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                with self.assertRaises(ValueError):
                    nninit.xavier_normal(tensor)

    def test_xavier_uniform(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for dims in [2, 4]:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                 as_variable=as_variable)
                    gain = 1

                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        nninit.xavier_uniform(input_tensor, gain=gain)
                    else:
                        nninit.xavier_uniform(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    tensor_shape = input_tensor.numpy().shape
                    receptive_field = np.prod(tensor_shape[2:])
                    expected_std = gain * np.sqrt(2.0 / ((tensor_shape[1] + tensor_shape[0]) * receptive_field))
                    bounds = expected_std * np.sqrt(3)
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    def test_xavier_normal(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for dims in [2, 4]:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                 as_variable=as_variable)
                    gain = 1

                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        nninit.xavier_normal(input_tensor, gain=gain)
                    else:
                        nninit.xavier_normal(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    tensor_shape = input_tensor.numpy().shape
                    receptive_field = np.prod(tensor_shape[2:])

                    expected_std = gain * np.sqrt(2.0 / ((tensor_shape[1] + tensor_shape[0]) * receptive_field))
                    assert self._is_normal(input_tensor, 0, expected_std)

    def test_kaiming_unifrom_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                    nninit.kaiming_uniform(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        for as_variable in [True, False]:
            for dims in [0, 1]:
                with self.assertRaises(ValueError):
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
                    nninit.kaiming_normal(tensor)

    def test_kaiming_uniform(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for dims in [2, 4]:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                 as_variable=as_variable)
                    receptive_field = np.prod(input_tensor.size()[2:])
                    gain = 1
                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        nninit.kaiming_uniform(input_tensor, gain=gain)
                    else:
                        nninit.kaiming_uniform(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    expected_std = gain * np.sqrt(1.0 / (input_tensor.size(1) * receptive_field))
                    bounds = expected_std * np.sqrt(3.0)
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    def test_kaiming_normal(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for dims in [2, 4]:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25,
                                                                 as_variable=as_variable)
                    receptive_field = np.prod(input_tensor.size()[2:])
                    gain = 1
                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        nninit.kaiming_normal(input_tensor, gain=gain)
                    else:
                        nninit.kaiming_normal(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    expected_std = gain * np.sqrt(1.0 / (input_tensor.size(1) * receptive_field))
                    assert self._is_normal(input_tensor, 0, expected_std)

    def test_sparse_only_works_on_2d_inputs(self):
        for as_variable in [True, False]:
            for dims in [1, 3]:
                with self.assertRaises(ValueError):
                    sparsity = self._random_float(0.1, 0.9)
                    tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3, as_variable=as_variable)
                    nninit.sparse(tensor, sparsity)

    def test_sparse_default_std(self):
        for as_variable in [True, False]:
            for use_random_std in [True, False]:
                input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35, as_variable=as_variable)
                rows, cols = input_tensor.size(0), input_tensor.size(1)
                sparsity = self._random_float(0.1, 0.2)

                std = 0.01  # default std
                if use_random_std:
                    std = self._random_float(0.01, 0.2)
                    nninit.sparse(input_tensor, sparsity=sparsity, std=std)
                else:
                    nninit.sparse(input_tensor, sparsity=sparsity)

                if as_variable:
                    input_tensor = input_tensor.data

                for col_idx in range(input_tensor.size(1)):
                    column = input_tensor[:, col_idx]
                    assert column[column == 0].nelement() >= np.ceil(sparsity * cols)

                assert self._is_normal(input_tensor[input_tensor != 0], 0, std)

    def test_orthogonal(self):
        for as_variable in [True, False]:
            for use_gain in [True, False]:
                for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                    input_tensor = torch.zeros(tensor_size)
                    gain = 1.0

                    if as_variable:
                        input_tensor = Variable(input_tensor)

                    if use_gain:
                        gain = self._random_float(0.1, 2)
                        nninit.orthogonal(input_tensor, gain=gain)
                    else:
                        nninit.orthogonal(input_tensor)

                    if as_variable:
                        input_tensor = input_tensor.data

                    rows, cols = tensor_size[0], int(np.prod(tensor_size[1:]))
                    flattened_tensor = input_tensor.view(rows, cols).numpy()
                    if rows > cols:
                        assert np.allclose(np.dot(flattened_tensor.T, flattened_tensor), np.eye(cols) * gain ** 2,
                                           atol=1e-6)
                    else:
                        assert np.allclose(np.dot(flattened_tensor, flattened_tensor.T), np.eye(rows) * gain ** 2,
                                           atol=1e-6)


if __name__ == "__main__":
    unittest.main()

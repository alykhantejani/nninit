from pytest import mark
import nninit
import torch
from numpy.random import randint, random_sample
import numpy as np
from scipy import stats


np.random.seed(123)
torch.manual_seed(123)


def _random_float(a, b):
    return (b - a) * random_sample() + a


def _create_random_nd_tensor(dims, size_min, size_max):
    size = [randint(size_min, size_max) for _ in range(dims)]
    return torch.from_numpy(np.ndarray(size)).zero_()


def _is_uniform(tensor, a, b):
    p_value = stats.kstest(tensor.numpy().flatten(), 'uniform', args=(a, (b - a))).pvalue
    return p_value > 0.01


def _is_normal(tensor, mean, std):
    p_value = stats.kstest(tensor.numpy().flatten(), 'norm', args=(mean, std)).pvalue
    return p_value > 0.01


@mark.parametrize("dims", [1, 2, 4])
def test_uniform(dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=50)
    a = _random_float(-3, 3)
    b = a + _random_float(1, 5)
    nninit.uniform(input_tensor, a=a, b=b)

    assert _is_uniform(input_tensor, a, b)


@mark.parametrize("dims", [1, 2, 4])
def test_normal(dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=50)
    mean = _random_float(-3, 3)
    std = _random_float(1, 5)
    nninit.normal(input_tensor, mean=mean, std=std)

    assert _is_normal(input_tensor, mean, std)


@mark.parametrize("dims", [1, 2, 4])
def test_constant(dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=1, size_max=5)
    val = _random_float(1, 10)
    nninit.constant(input_tensor, val)
    assert np.allclose(input_tensor.numpy(), input_tensor.clone().fill_(val).numpy())


@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def test_xavier_uniform(use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=35)
    gain = 1

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.xavier_uniform(input_tensor, gain=gain)
    else:
        nninit.xavier_uniform(input_tensor)

    tensor_shape = input_tensor.numpy().shape
    receptive_field = np.prod(tensor_shape[2:])
    expected_std = gain * np.sqrt(2.0 / ((tensor_shape[1] + tensor_shape[0]) * receptive_field))
    bounds = expected_std * np.sqrt(3)
    assert _is_uniform(input_tensor, -bounds, bounds)
    assert np.allclose(input_tensor.std(), expected_std, atol=1e-2)


@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def test_xavier_normal(use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=35)
    gain = 1

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.xavier_normal(input_tensor, gain=gain)
    else:
        nninit.xavier_normal(input_tensor)

    tensor_shape = input_tensor.numpy().shape
    receptive_field = np.prod(tensor_shape[2:])

    expected_std = gain * np.sqrt(2.0 / ((tensor_shape[1] + tensor_shape[0]) * receptive_field))
    assert _is_normal(input_tensor, 0, expected_std)


@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def kaiming_uniform(use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=35)
    tensor_shape = input_tensor.numpy().shape
    receptive_field = np.prod(tensor_shape[2:])

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.kaiming_uniform(input_tensor, gain=gain)
    else:
        nninit.kaiming_uniform(input_tensor)

    expected_std = gain * np.sqrt(1.0 / (tensor_shape[1] * receptive_field))
    bounds = expected_std * np.sqrt(3.0)
    assert _is_uniform(input_tensor, -bounds, bounds)
    assert np.allclose(input_tensor.std(), expected_std, atol=1e-2)


@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def kaiming_normal(use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=35)
    tensor_shape = input_tensor.numpy().shape
    receptive_field = np.prod(tensor_shape[2:])

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.kaiming_normal(input_tensor, gain=gain)
    else:
        nninit.kaiming_normal(input_tensor)

    expected_std = gain * np.sqrt(1.0 / (tensor_shape[1] * receptive_field))
    assert _is_normal(input_tensor, 0, expected_std)

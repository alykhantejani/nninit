from pytest import mark
import nninit
import torch
from numpy.random import randint, random_sample
import numpy as np
from scipy import stats


def _random_float(a, b):
    return (b - a) * random_sample() + a


# TODO: set random seed at beggining of tests

@mark.parametrize("dims", [1, 2, 4])
def test_uniform(dims):
    np.random.seed(123)
    torch.manual_seed(123)
    sizes = [randint(30, 50) for _ in range(dims)]
    input_tensor = torch.from_numpy(np.ndarray(sizes)).zero_()

    a = _random_float(-3, 3)
    b = a + _random_float(1, 5)
    nninit.uniform(input_tensor, a=a, b=b)

    p_value = stats.kstest(input_tensor.numpy().flatten(), 'uniform', args=(a, (b - a))).pvalue
    assert p_value > 0.05


@mark.parametrize("dims", [1, 2, 4])
def test_normal(dims):
    np.random.seed(123)
    torch.manual_seed(123)

    sizes = [randint(30, 50) for _ in range(dims)]
    input_tensor = torch.from_numpy(np.ndarray(sizes)).zero_()

    mean = _random_float(-3, 3)
    std = _random_float(1, 5)
    nninit.normal(input_tensor, mean=mean, std=std)

    p_value = stats.kstest(input_tensor.numpy().flatten(), 'norm', args=(mean, std)).pvalue
    assert p_value > 0.05


@mark.parametrize("dims", [1, 2, 4])
def test_constant(dims):
    sizes = [randint(1, 5) for _ in dims]
    input_tensor = torch.from_numpy(np.ndarray(sizes)).zero_()
    val = _random_float(1, 10)
    nninit.constant(input_tensor, val)
    assert np.allclose(input_tensor.numpy(), input_tensor.clone().fill_(val).numpy())

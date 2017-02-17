import pytest
from pytest import mark
import math
import nninit
import torch
from numpy.random import randint
import numpy as np
from scipy import stats


@mark.parametrize("dims", [1, 2, 4])
def test_uniform(dims):
    np.random.seed(123)
    torch.manual_seed(123)
    sizes = [randint(30, 50) for _ in range(dims)]
    input_tensor = torch.from_numpy(np.ndarray(sizes)).zero_()

    a = randint(-3, 3)
    b = a + randint(1, 5)
    nninit.uniform(input_tensor, a=a, b=b)

    p_value = stats.kstest(input_tensor.numpy().flatten(), 'uniform', args=(a, (b - a))).pvalue
    assert p_value > 0.05


@mark.parametrize("dims", [1, 2, 4])
def test_normal(dims):
    np.random.seed(123)
    torch.manual_seed(123)

    sizes = [randint(30, 50) for _ in range(dims)]
    input_tensor = torch.from_numpy(np.ndarray(sizes)).zero_()

    mean = randint(-3, 3)
    std = randint(1, 5)
    nninit.normal(input_tensor, mean=mean, std=std)

    p_value = stats.kstest(input_tensor.numpy().flatten(), 'norm', args=(mean, std)).pvalue
    assert p_value > 0.05

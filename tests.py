from pytest import mark
from pytest import raises
import nninit
import torch
from numpy.random import randint, random_sample
import numpy as np
from scipy import stats
from torch.autograd import Variable

np.random.seed(123)
torch.manual_seed(123)


def _random_float(a, b):
    return (b - a) * random_sample() + a


def _create_random_nd_tensor(dims, size_min, size_max, as_variable):
    size = [randint(size_min, size_max + 1) for _ in range(dims)]
    tensor = torch.zeros(size)
    if as_variable:
        tensor = Variable(tensor)
    return tensor


def _is_uniform(tensor, a, b):
    if isinstance(tensor, Variable):
        tensor = tensor.data
    p_value = stats.kstest(tensor.numpy().flatten(), 'uniform', args=(a, (b - a))).pvalue
    return p_value > 0.0001


def _is_normal(tensor, mean, std):
    if isinstance(tensor, Variable):
        tensor = tensor.data
    p_value = stats.kstest(tensor.numpy().flatten(), 'norm', args=(mean, std)).pvalue
    return p_value > 0.0001


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [1, 2, 4])
def test_uniform(as_variable, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)

    a = _random_float(-3, 3)
    b = a + _random_float(1, 5)
    nninit.uniform(input_tensor, a=a, b=b)

    assert _is_uniform(input_tensor, a, b)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [1, 2, 4])
def test_normal(as_variable, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=30, size_max=50, as_variable=as_variable)
    mean = _random_float(-3, 3)
    std = _random_float(1, 5)
    nninit.normal(input_tensor, mean=mean, std=std)

    assert _is_normal(input_tensor, mean, std)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [1, 2, 4])
def test_constant(as_variable, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=1, size_max=5, as_variable=as_variable)
    val = _random_float(1, 10)
    nninit.constant(input_tensor, val)
    if as_variable:
        input_tensor = input_tensor.data

    assert np.allclose(input_tensor.numpy(), input_tensor.clone().fill_(val).numpy())


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [0, 1])
def test_xavier_uniform_errors_on_inputs_smaller_than_2d(as_variable, dims):
    tensor = _create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
    with raises(ValueError):
        nninit.xavier_uniform(tensor)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [0, 1])
def test_xavier_normal_errors_on_inputs_smaller_than_2d(as_variable, dims):
    tensor = _create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable)
    with raises(ValueError):
        nninit.xavier_normal(tensor)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def test_xavier_uniform(as_variable, use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=20, size_max=25, as_variable=as_variable)
    gain = 1

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.xavier_uniform(input_tensor, gain=gain)
    else:
        nninit.xavier_uniform(input_tensor)

    if as_variable:
        input_tensor = input_tensor.data

    tensor_shape = input_tensor.numpy().shape
    receptive_field = np.prod(tensor_shape[2:])
    expected_std = gain * np.sqrt(2.0 / ((tensor_shape[1] + tensor_shape[0]) * receptive_field))
    bounds = expected_std * np.sqrt(3)
    assert _is_uniform(input_tensor, -bounds, bounds)
    assert np.allclose(input_tensor.std(), expected_std, atol=1e-2)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def test_xavier_normal(as_variable, use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=20, size_max=25, as_variable=as_variable)
    gain = 1

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.xavier_normal(input_tensor, gain=gain)
    else:
        nninit.xavier_normal(input_tensor)

    if as_variable:
        input_tensor = input_tensor.data

    tensor_shape = input_tensor.numpy().shape
    receptive_field = np.prod(tensor_shape[2:])

    expected_std = gain * np.sqrt(2.0 / ((tensor_shape[1] + tensor_shape[0]) * receptive_field))
    assert _is_normal(input_tensor, 0, expected_std)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [0, 1])
def test_kaiming_unifrom_errors_on_inputs_smaller_than_2d(as_variable, dims):
    with raises(ValueError):
        nninit.kaiming_uniform(_create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable))


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [0, 1])
def test_kaiming_normal_errors_on_inputs_smaller_than_2d(as_variable, dims):
    with raises(ValueError):
        nninit.kaiming_normal(_create_random_nd_tensor(dims, size_min=1, size_max=1, as_variable=as_variable))


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def test_kaiming_uniform(as_variable, use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=20, size_max=25, as_variable=as_variable)
    receptive_field = np.prod(input_tensor.size()[2:])
    gain = 1
    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.kaiming_uniform(input_tensor, gain=gain)
    else:
        nninit.kaiming_uniform(input_tensor)

    if as_variable:
        input_tensor = input_tensor.data

    expected_std = gain * np.sqrt(1.0 / (input_tensor.size(1) * receptive_field))
    bounds = expected_std * np.sqrt(3.0)
    assert _is_uniform(input_tensor, -bounds, bounds)
    assert np.allclose(input_tensor.std(), expected_std, atol=1e-2)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("use_gain", [True, False])
@mark.parametrize("dims", [2, 4])
def test_kaiming_normal(as_variable, use_gain, dims):
    input_tensor = _create_random_nd_tensor(dims, size_min=20, size_max=25, as_variable=as_variable)
    receptive_field = np.prod(input_tensor.size()[2:])
    gain = 1
    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.kaiming_normal(input_tensor, gain=gain)
    else:
        nninit.kaiming_normal(input_tensor)

    if as_variable:
        input_tensor = input_tensor.data

    expected_std = gain * np.sqrt(1.0 / (input_tensor.size(1) * receptive_field))
    assert _is_normal(input_tensor, 0, expected_std)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("dims", [1, 3])
def test_sparse_only_works_on_2d_inputs(as_variable, dims):
    with raises(ValueError):
        sparsity = _random_float(0.1, 0.9)
        nninit.sparse(_create_random_nd_tensor(dims, size_min=1, size_max=3, as_variable=as_variable), sparsity)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("use_random_std", [True, False])
def test_sparse_default_std(as_variable, use_random_std):
    input_tensor = _create_random_nd_tensor(2, size_min=30, size_max=35, as_variable=as_variable)
    rows, cols = input_tensor.size(0), input_tensor.size(1)
    sparsity = _random_float(0.1, 0.2)

    std = 0.01  # default std
    if use_random_std:
        std = _random_float(0.01, 0.2)
        nninit.sparse(input_tensor, sparsity=sparsity, std=std)
    else:
        nninit.sparse(input_tensor, sparsity=sparsity)

    if as_variable:
        input_tensor = input_tensor.data

    for col_idx in range(input_tensor.size(1)):
        column = input_tensor[:, col_idx]
        assert column[column == 0].nelement() >= np.ceil(sparsity * cols)

    assert _is_normal(input_tensor[input_tensor != 0], 0, std)


@mark.parametrize("as_variable", [True, False])
@mark.parametrize("use_gain", [True, False])
@mark.parametrize("tensor_size", [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]])
def test_orthogonal(as_variable, use_gain, tensor_size):
    input_tensor = torch.zeros(tensor_size)
    gain = 1.0

    if as_variable:
        input_tensor = Variable(input_tensor)

    if use_gain:
        gain = _random_float(0.1, 2)
        nninit.orthogonal(input_tensor, gain=gain)
    else:
        nninit.orthogonal(input_tensor)

    if as_variable:
        input_tensor = input_tensor.data

    rows, cols = tensor_size[0], int(np.prod(tensor_size[1:]))
    flattened_tensor = input_tensor.view(rows, cols).numpy()
    if rows > cols:
        assert np.allclose(np.dot(flattened_tensor.T, flattened_tensor), np.eye(cols) * gain ** 2, atol=1e-6)
    else:
        assert np.allclose(np.dot(flattened_tensor, flattened_tensor.T), np.eye(rows) * gain ** 2, atol=1e-6)

import numpy as np


def uniform(tensor, a=0, b=1):
    return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    return tensor.normal_(mean, std)


def constant(tensor, val):
    return tensor.fill_(val)


def _calculate_fan_in_and_fan_out(tensor):
    if len(tensor.size()) < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if len(tensor.size()) == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def he_uniform(tensor, gain=1):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(1.0 / fan_in)
    a = np.sqrt(3.0) * std
    return tensor.uniform_(-a, a)


def he_normal(tensor, gain=1):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(1.0 / fan_in)
    return tensor.normal_(0, std)
import numpy as np


def uniform(tensor, a=0, b=1):
    return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    return tensor.normal_(mean, std)


def constant(tensor, val):
    return tensor.fill_(val)


def _calculate_fan_in_and_fan_out(tensor):
    if len(tensor.size()) == 2: # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    elif len(tensor.size()) == 4: # Conv2D
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        kernel_height = tensor.size(2)
        kernel_width = tensor.size(3)
        fan_in = num_input_fmaps * kernel_height * kernel_width
        fan_out = num_output_fmaps * kernel_height * kernel_width
    else:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())
    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0/ (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0/ (fan_in + fan_out))
    return tensor.normal_(mean=0, var=std**2)


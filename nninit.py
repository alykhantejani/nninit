import numpy as np


def uniform(tensor, a=0, b=1):
    """Fills the input tensor with values drawn from a uniform U(a,b)

    Args:
        tensor: a n-dimension torch.Tensor
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.uniform(w)
    """
    return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    """Fills the input tensor with values drawn from a normal distribution with the given mean and std

    Args:
        tensor: a n-dimension torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.normal(w)
    """
    return tensor.normal_(mean, std)


def constant(tensor, val):
    """Fills the input tensor with the value `val`

    Args:
        tensor: a n-dimension torch.Tensor
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.constant(w)
    """
    return tensor.fill_(val)


def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
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
    """Fills the input tensor with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a uniform distribution.

       The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(2/(fan_in + fan_out))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: optional gain to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_uniform(w, gain=np.sqrt(2.0))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input tensor with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal distribution.

       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(2/(fan_in + fan_out))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: optional gain to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_normal(w, gain=np.sqrt(2.0))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def kaiming_uniform(tensor, gain=1):
    """Fills the input tensor with values according to the method described in "Delving deep into rectifiers: Surpassing
       human-level performance on ImageNet classification" - He, K. et al using a normal distribution.

       The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(1/(fan_in))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: optional gain to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.kaiming_uniform(w, gain=np.sqrt(2.0))
    """
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(1.0 / fan_in)
    a = np.sqrt(3.0) * std
    return tensor.uniform_(-a, a)


def kaiming_normal(tensor, gain=1):
    """Fills the input tensor with values according to the method described in "Delving deep into rectifiers: Surpassing
       human-level performance on ImageNet classification" - He, K. et al using a normal distribution.

       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(1/(fan_in))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: optional gain to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.kaiming_normal(w, gain=np.sqrt(2.0))
    """

    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(1.0 / fan_in)
    return tensor.normal_(0, std)


def sparse(tensor, sparsity, std=0.01):
    """Fills the 2D input tensor as a sparse matrix, where the non-zero elements will be drawn from a normal distribution
     with mean=0 and std=`std`.
     Reference: "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al.

    Args:
        tensor: a n-dimension torch.Tensor
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate the non-zero values

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.sparse(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Sparse initialization only supported for 2D inputs")
    tensor.normal_(0, std)
    rows, cols = tensor.size(0), tensor.size(1)
    num_zeros = int(np.ceil(cols * sparsity))

    for col_idx in range(tensor.size(1)):
        row_indices = np.arange(rows)
        np.random.shuffle(row_indices)
        zero_indices = row_indices[:num_zeros]
        tensor.numpy()[zero_indices, col_idx] = 0

    return tensor

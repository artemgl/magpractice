from multipledispatch import dispatch
from math import pi, log
import matplotlib.pyplot as plt
import torch


def plot_fun(fun, min_arg=None, max_arg=None):
    """
    Plot the function with matplotlib.
    :param fun: The function to plot.
    :param min_arg: Left edge of domain.
    :param max_arg: Right edge of domain.
    """
    if min_arg is None:
        min_arg = fun.min_arg
    if max_arg is None:
        max_arg = fun.max_arg
    k = 1000
    args = [min_arg + (max_arg - min_arg) * i / k for i in range(k)]
    vals = fun(torch.Tensor(args)).type(torch.complex128).cpu()
    plt.plot(args, torch.real(vals), label='Real')
    plt.plot(args, torch.imag(vals), label='Imaginary')
    plt.legend(loc='upper right')
    plt.show()


def complex_pow(a, b):
    return torch.where(torch.logical_and(torch.abs(a) == 0, b == 0), 1, torch.pow(a, b))


@dispatch(torch.Tensor, torch.Tensor, torch.Tensor)
def hermite_H(x, y, n):
    """
    Generalized Hermite polynomial.
    :param x: First argument of the polynomial.
    :param y: Second argument of the polynomial.
    :param n: Index of the polynomial.
    """
    k = torch.arange(torch.max(n)//2 + 1).view(-1, 1, 1, 1)
    x = x.view(1, -1, 1, 1)
    y = y.view(1, 1, -1, 1)
    n = n.view(1, 1, 1, -1)
    return torch.where(k > n//2, 0, torch.exp(torch.lgamma(n + 1) - torch.lgamma(n - 2*k + 1) - torch.lgamma(k + 1)) * complex_pow(x, n - 2*k) * complex_pow(y, k)).sum(0).view(x.numel(), y.numel(), n.numel())


@dispatch(torch.Tensor, torch.Tensor)
def hermite_H(x, n):
    """
    Hermite polynomial.
    :param x: Argument of the polynomial.
    :param n: Index of the polynomial.
    """
    k = torch.arange(torch.max(n)//2 + 1).view(-1, 1, 1)
    x = x.view(1, -1, 1)
    n = n.view(1, 1, -1)
    return torch.where(k > n//2, 0, torch.exp(torch.lgamma(n + 1) - torch.lgamma(n - 2*k + 1) - torch.lgamma(k + 1)) * complex_pow(2*x, n - 2*k) * (-1)**k).sum(0).view(x.numel(), n.numel())


def hermite_fun(x, n):
    """
    Hermite function.
    :param x: Argument of the function.
    :param n: Index of the function.
    """
    return torch.exp(-0.5*(x*x + n*log(2) + torch.lgamma(n + 1) + 0.5*log(pi))) * torch.special.hermite_polynomial_h(x, n)



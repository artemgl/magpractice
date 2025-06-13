import torch
import special
from math import pi, log
from torchquad import MonteCarlo, Gaussian, Trapezoid, Simpson, Boole


class WaveFunStep:
    """
    Wave function represented in step form. Outside the domain the function value is considered to be zero.
    :param values: Values of a function at evenly distributed points over its domain.
    :param min_arg: Left edge of domain.
    :param max_arg: Right edge of domain.
    """
    def __init__(self, values, min_arg, max_arg):
        self.values = values.flatten()
        self.min_arg = min_arg
        self.max_arg = max_arg

    def __call__(self, args):
        shape = args.shape
        indices = torch.round((torch.numel(self.values) - 1) * (args - self.min_arg) / (self.max_arg - self.min_arg))
        condition = torch.logical_or(indices < 0, indices >= torch.numel(self.values))
        indices[condition] = 0
        return torch.where(condition, 0, torch.gather(self.values, 0, indices.flatten().long()).view(shape))

    def norm(self, integral_density):
        mc = Boole()
        norm = mc.integrate(
            lambda x: self(x) * torch.conj(self(x)),
            dim=1,
            N=round(integral_density * (self.max_arg - self.min_arg)),
            integration_domain=[[self.min_arg, self.max_arg]],
            backend="torch",
        )

        return torch.sqrt(norm)

    def normalize(self, integral_density=20):
        norm = self.norm(integral_density)
        self.values /= norm
        return norm

    def derivative(self):
        step = (self.max_arg - self.min_arg) / (torch.numel(self.values) - 1)
        deriv_values = (torch.cat((self.values, torch.zeros(2)), 0) - torch.cat((torch.zeros(2), self.values), 0)) / (
                    2 * step)
        return WaveFunStep(deriv_values, self.min_arg - step, self.max_arg + step)

    def derivative2(self):
        step = (self.max_arg - self.min_arg) / (torch.numel(self.values) - 1)
        deriv_values = (torch.cat((self.values, torch.zeros(2)), 0) - 2 * torch.cat(
            (torch.zeros(1), self.values, torch.zeros(1)), 0) + torch.cat((torch.zeros(2), self.values), 0)) / (
                                   step * step)
        return WaveFunStep(deriv_values, self.min_arg - step, self.max_arg + step)


class WaveFunContinuous:
    """
    Wave function represented in continuous form.
    :param fun: Explicit function definition.
    :param min_arg: Left edge of domain.
    :param max_arg: Right edge of domain.
    """
    def __init__(self, fun, min_arg, max_arg):
        self.fun = fun
        self.min_arg = min_arg
        self.max_arg = max_arg

    def __call__(self, args):
        return self.fun(args)


class WaveFunHermite:
    """
    Wave function represented in series of Gauss-Hermite functions.
    :param coeffs: Expansion coefficients of a function with respect to the Gauss-Hermite basis.
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, args):
        n = self.coeffs.numel()
        k = torch.arange(n)
        return special.hermite_fun(args.view(-1, 1), k).to(torch.complex128) @ self.coeffs

    def integrate(self):
        """ Integrate the function over the entire real axis. """
        k = torch.arange(torch.numel(self.coeffs))
        integrals = torch.where(k%2 == 0, pi**0.25 * torch.exp(0.5*(torch.lgamma(k + 1) - (k - 1)*log(2)) - torch.lgamma(k/2 + 1)), 0).to(torch.complex128)
        return integrals @ self.coeffs
from math import pi, sin, cos, sinh, cosh, tanh, factorial, pow, exp, sqrt, log, asin, acos
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import numpy as np
from torchquad import MonteCarlo, Gaussian, Trapezoid, Simpson, Boole, set_up_backend
import warnings
warnings.filterwarnings('ignore')

device = "cpu"
if device == 'cuda':
  set_up_backend("torch", data_type="float64")

class WaveFunContinious:
  def __init__(self, fun, min_arg, max_arg):
    self.fun = fun
    self.min_arg = min_arg
    self.max_arg = max_arg

  def __call__(self, args):
    return self.fun(args)

class WaveFunStep:
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

  def diff(self, other, integral_density):
    min_arg = min(self.min_arg, other.min_arg)
    max_arg = min(self.max_arg, other.max_arg)

    mc = Boole()
    res = mc.integrate(
        lambda x: abs(self(x) - other(x)) ** 2,
        dim=1,
        N=round(integral_density * (max_arg - min_arg)),
        integration_domain=[[min_arg, max_arg]],
        backend="torch",
    )

    return torch.sqrt(res)

  def derivative(self):
    step = (self.max_arg - self.min_arg) / (torch.numel(self.values) - 1)
    deriv_values = (torch.cat((self.values, torch.zeros(2)), 0) - torch.cat((torch.zeros(2), self.values), 0)) / (2 * step)
    return WaveFunStep(deriv_values, self.min_arg - step, self.max_arg + step)

  def derivative2(self):
    step = (self.max_arg - self.min_arg) / (torch.numel(self.values) - 1)
    deriv_values = (torch.cat((self.values, torch.zeros(2)), 0) - 2 * torch.cat((torch.zeros(1), self.values, torch.zeros(1)), 0) + torch.cat((torch.zeros(2), self.values), 0)) / (step * step)
    return WaveFunStep(deriv_values, self.min_arg - step, self.max_arg + step)

def dot_product(n, x):
  return torch.exp(-x * x / 2) * torch.special.hermite_polynomial_h(x, n) / (pow(pi, 1 / 4) * pow(2, n / 2) * sqrt(factorial(n)))

def compute_output(input_fun1, input_fun2, n_measured_photons, bs_angle, output_density, integral_density):
  rho = cos(bs_angle)
  t = sin(bs_angle)

  # min_arg = -20
  # max_arg = 20
  min_arg = input_fun1.min_arg * rho + input_fun2.min_arg * t
  max_arg = input_fun1.max_arg * rho + input_fun2.max_arg * t
  x_2 = torch.arange(min_arg, max_arg, 1 / output_density, device=device)

  def fun(x):
    x = x.view(-1, 1)
    # return input_fun1(t * x + rho * x_2) * input_fun2(-rho * x + t * x_2)
    return input_fun1(t * x + rho * x_2) * input_fun2(-rho * x + t * x_2) * dot_product(n_measured_photons, x)

  mc = Boole()
  # mc = MonteCarlo()
  result = mc.integrate(
      fun,
      dim=1,
      N=round(integral_density * (t * (input_fun1.max_arg - input_fun1.min_arg) + rho * (input_fun2.max_arg - input_fun2.min_arg))),
      integration_domain=[[t * input_fun1.min_arg - rho * input_fun2.max_arg, t * input_fun1.max_arg - rho * input_fun2.min_arg]],
      # N=round(integral_density * 40),
      # integration_domain=[[-20, 20]],
      backend="torch",
  )

  return WaveFunStep(result, min_arg, max_arg)

def mean_O(input_fun, gamma, integral_density):
  fun_derivative = input_fun.derivative()
  min_arg = fun_derivative.min_arg
  max_arg = fun_derivative.max_arg

  def fun(x):
    x = x.view(-1, 1)
    return input_fun(x).conj() * (-1j * fun_derivative(x) + gamma * x * x * input_fun(x))

  mc = Boole()
  result = mc.integrate(
      fun,
      dim=1,
      N=round(integral_density * (max_arg - min_arg)),
      integration_domain=[[min_arg, max_arg]],
      backend="torch",
  )

  return result

def mean_O_squared(input_fun, gamma, integral_density):
  fun_derivative = input_fun.derivative()
  fun_derivative2 = input_fun.derivative2()
  min_arg = min(fun_derivative.min_arg, fun_derivative2.min_arg)
  max_arg = max(fun_derivative.max_arg, fun_derivative2.max_arg)

  def fun(x):
    x = x.view(-1, 1)
    return input_fun(x).conj() * (-fun_derivative2(x) + gamma * x * ((-2j + gamma * x * x * x) * input_fun(x) - 2j * x * fun_derivative(x)))

  mc = Boole()
  result = mc.integrate(
      fun,
      dim=1,
      N=round(integral_density * (max_arg - min_arg)),
      integration_domain=[[min_arg, max_arg]],
      backend="torch",
  )

  return result

def nonlinear_compression(input_fun, gamma, integral_density):
  return torch.real(mean_O_squared(input_fun, gamma, integral_density) - mean_O(input_fun, gamma, integral_density)**2) * 2 / 3 * (2 / gamma)**(2 / 3)


if __name__ == '__main__':
    def loss(params1, params2, angle):
        # Инициализация функций
        A = 10
        r1 = params1[0]
        alpha1_real = params1[2]
        alpha1_imag = params1[3]
        fun1 = WaveFunContinious(lambda x: torch.exp(
            1j * sqrt(2) * x * alpha1_imag - (x - sqrt(2) * alpha1_real) ** 2 / 2 / (
                        cosh(2 * r1) - sinh(2 * r1))) / pow(pi * (cosh(2 * r1) - sinh(2 * r1)), 1 / 4), -A / sqrt(2),
                                 A / sqrt(2))

        r2 = params2[0]
        phi = params2[1]
        alpha2_real = params2[2]
        alpha2_imag = params2[3]
        fun2 = WaveFunContinious(lambda x: torch.exp(
            1j * sqrt(2) * x * alpha2_imag - (x - sqrt(2) * alpha2_real) ** 2 / 2 * (
                        1 + 1j * sin(phi) * sinh(2 * r2)) / (cosh(2 * r2) - cos(phi) * sinh(2 * r2))) / pow(
            pi * (cosh(2 * r2) - cos(phi) * sinh(2 * r2)), 1 / 4), -A / sqrt(2), A / sqrt(2))

        # Выходная функций, измерили 1 фотон
        fun = compute_output(fun1, fun2, 1, angle, 30, 15)

        # Normalize
        vals = fun.values / fun.norm(15)
        gun = WaveFunStep(vals, -A, A)

        # gamma = 1
        return nonlinear_compression(gun, 1, 15)


    params1 = torch.rand(4, requires_grad=True, dtype=torch.float64)
    params2 = torch.rand(4, requires_grad=True, dtype=torch.float64)
    angle = torch.rand(1, requires_grad=True, dtype=torch.float64)

    loss_history = []

    optimizer = torch.optim.Adam([params1, params2, angle], lr=0.1, maximize=False)
    n_epochs = 100
    for epoch_index in range(n_epochs):
        optimizer.zero_grad()
        loss_value = loss(params1, params2, angle[0])
        loss_history.append(loss_value.data)
        loss_value.backward()
        optimizer.step()

    print(loss_value)

    # plt.plot(loss_history, label='Loss')
    # plt.legend(loc='center right')

    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("graph.pdf", format="pdf", bbox_inches="tight")
    plt.show()
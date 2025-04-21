from math import pi, sin, cos, sinh, cosh, tanh, factorial, pow, exp, sqrt, log, asin, acos
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import mpmath
import numpy as np

device = "cuda" # @param ["cpu", "cuda"]
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

def plot_fun(fun, min_arg=None, max_arg=None):
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

def diff(fun1, fun2, min_arg=None, max_arg=None):
  if min_arg is None:
    min_arg = fun1.min_arg
  if max_arg is None:
    max_arg = fun1.max_arg
  k = 1000
  args = [min_arg + (max_arg - min_arg) * i / k for i in range(k)]
  vals1 = fun1(torch.Tensor(args)).cpu()
  vals2 = fun2(torch.Tensor(args)).cpu()

  return ((vals1 - vals2) * (vals1 - vals2).conj()).mean()

class WaveFunContinious:
  def __init__(self, fun, min_arg, max_arg):
    self.fun = fun
    self.min_arg = min_arg
    self.max_arg = max_arg

  def __call__(self, args):
    return self.fun(args)

def complex_pow(a, b):
  return torch.where(torch.logical_and(torch.abs(a) == 0, b == 0), 1, torch.pow(a, b))

# Сделать перегрузку
def hermite_H2(x, y, n):
  k = torch.arange(torch.max(n)//2 + 1).view(-1, 1, 1, 1)
  x = x.view(1, -1, 1, 1)
  y = y.view(1, 1, -1, 1)
  n = n.view(1, 1, 1, -1)
  return torch.where(k > n//2, 0, torch.exp(torch.lgamma(n + 1) - torch.lgamma(n - 2*k + 1) - torch.lgamma(k + 1)) * complex_pow(x, n - 2*k) * complex_pow(y, k)).sum(0).view(x.numel(), y.numel(), n.numel())

def hermite_H(x, n):
  k = torch.arange(torch.max(n)//2 + 1).view(-1, 1, 1)
  x = x.view(1, -1, 1)
  n = n.view(1, 1, -1)
  return torch.where(k > n//2, 0, torch.exp(torch.lgamma(n + 1) - torch.lgamma(n - 2*k + 1) - torch.lgamma(k + 1)) * complex_pow(2*x, n - 2*k) * (-1)**k).sum(0).view(x.numel(), n.numel())

def hermite_fun(x, n):
  # return torch.exp(-x * x / 2) * torch.special.hermite_polynomial_h(x, n) / torch.pow(2**n * torch.exp(torch.lgamma(n + 1)) * sqrt(pi), 0.5)
  return torch.exp(-0.5*(x*x + n*log(2) + torch.lgamma(n + 1) + 0.5*log(pi))) * torch.special.hermite_polynomial_h(x, n)

class WaveFunHermite:
  def __init__(self, coeffs):
    self.coeffs = coeffs

  def __call__(self, args):
    n = self.coeffs.numel()
    k = torch.arange(n)
    return hermite_fun(args.view(-1, 1), k).to(torch.complex128) @ self.coeffs

  def integrate(self):
    k = torch.arange(torch.numel(self.coeffs))
    integrals = torch.where(k%2 == 0, pi**0.25 * torch.exp(0.5*(torch.lgamma(k + 1) - (k - 1)*log(2)) - torch.lgamma(k/2 + 1)), 0).to(torch.complex128)
    return integrals @ self.coeffs

class Compresser():
  def __init__(self, n, n_points):
    with open('points.npy', 'rb') as f:
      self.points = torch.tensor(np.load(f))
    with open('coeffs.npy', 'rb') as f:
      self.coeffs = torch.tensor(np.load(f))
    with open('offset.npy', 'rb') as f:
      self.offset = torch.tensor(np.load(f))

  def __call__(self, arg):
    arg = arg.squeeze()
    return self.offset + (self.coeffs @ torch.nn.functional.relu(arg - self.points[:-1]))

n = 51
n_points = 1000

compress = Compresser(n, n_points)

with open('I.npy', 'rb') as f:
  I = torch.tensor(np.load(f))
with open('J.npy', 'rb') as f:
  J = torch.tensor(np.load(f))

def entangle_bs(fun1, fun2, angle, n_measured_photons, I, J):
  # Сколькими базисными функциями приближаем
  n = torch.numel(fun1.coeffs)

  # f(x) -> f(x + y)
  coeffs2d1 = torch.matmul(J, fun1.coeffs.view(-1, 1)).view(n, n)
  coeffs2d2 = torch.matmul(J, fun2.coeffs.view(-1, 1)).view(n, n)

  # Сжать по осям
  # f(x + y), g(x + y) -> f(t x + rho y), g(-rho x + t y)
  # st = stretch(torch.sin(angle), n).to(torch.complex128)
  # rhot = stretch(torch.cos(angle), n).to(torch.complex128)
  st = compress(torch.sin(angle)).view(n, n).to(torch.complex128)
  rhot = compress(torch.cos(angle)).view(n, n).to(torch.complex128)
  reflect = torch.diag((-1)**torch.arange(n)).to(torch.complex128)
  coeffs2d1 = st @ coeffs2d1 @ rhot.T
  coeffs2d2 = rhot @ reflect @ coeffs2d2 @ st.T

  # Произведение функций
  a = torch.matmul(I, coeffs2d1).view(n, 1, n, n)
  b = torch.matmul(coeffs2d2, I).view(n, 1, n, n)
  product = torch.nn.functional.conv2d(a, b).view(n, n)

  # Измерили n_measured_photons фотонов
  c = product[n_measured_photons]

  return WaveFunHermite(c)

def mean_operator(fun, op):
  return fun.coeffs.conj() @ op @ fun.coeffs

# Оператор nxn производной
def compute_ddx(n):
  k = torch.arange(1, n)
  d = torch.sqrt(k/2)
  return torch.diag(d, 1) - torch.diag(d, -1)

# Оператор nxn умножения на x
def compute_x(n):
  k = torch.arange(1, n)
  d = torch.sqrt(k/2)
  return torch.diag(d, 1) + torch.diag(d, -1)

# Оператор nxn O(gamma)
def compute_O(gamma, n):
  d = compute_ddx(n)
  x = compute_x(n)
  return -1j * d + gamma * (x @ x)

def qubic_phase(fun, gamma):
  # .to(torch.complex128)? Оптимизировать
  O = compute_O(gamma, torch.numel(fun.coeffs)).to(torch.complex128)
  return (mean_operator(fun, O @ O) - mean_operator(fun, O)**2) / (1.5*(gamma/2)**(2/3))

def build_wave_fun(params, n):
  k = torch.arange(n)

  r = params[0]
  phi = params[1]
  b1 = params[2]
  b2 = params[3]

  b = b1 + 1j*b2

  a = torch.exp(1j*phi)*torch.tanh(r)
  c = torch.exp(2*r)*torch.sin(phi/2)**2 + torch.exp(-2*r)*torch.cos(phi/2)**2

  return torch.sqrt(1 - a) * torch.exp((1 - a) * b**2 / 4 - b1**2 * c / 2 - 0.25*torch.log(c) - 0.5*(k*log(2) + torch.lgamma(k + 1))) * hermite_H2((1 - a)*b, -a, k).view(-1)


if __name__ == '__main__':
    # n_attempts = 200
    # gammas = np.arange(0.1, 3, 0.1)
    gammas = np.array([1.0, 2.0])
    # gammas = np.array([0.2, 1.0, 1.5, 2.0])
    # gammas = np.array([1.0])
    # print(gammas)

    qp_history = []
    params_history = []
    for gamma in gammas:
        def forward(params1, params2, angle1):
            # Инициализация
            n = 51
            fun1 = WaveFunHermite(build_wave_fun(params1, n))
            fun2 = WaveFunHermite(build_wave_fun(params2, n))

            # Первая итерация, измерили 4 фотона
            # TODO: не передавать I, J как параметры
            fun = entangle_bs(fun1, fun2, angle1, 4, I, J)
            # Normalize
            coeffs = fun.coeffs / torch.sqrt(fun.coeffs.conj() @ fun.coeffs)
            fun = WaveFunHermite(coeffs)

            return torch.real(qubic_phase(fun, gamma))
        def loss(qubic_phase, params1, params2, angle1):
            return qubic_phase

        best_qp = 1000
        best_params = []
        loss_history = []
        # for attempt in range(n_attempts):
        while 10 * log(best_qp) / log(10) > -3.2:
        # for _ in range(1):
            params1 = torch.rand(4, requires_grad=True, dtype=torch.float64)
            params2 = torch.rand(4, requires_grad=True, dtype=torch.float64)
            # params3 = torch.rand(4, requires_grad=True, dtype=torch.float64)
            # params4 = torch.rand(4, requires_grad=True, dtype=torch.float64)
            angle1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
            # angle2 = torch.rand(1, requires_grad=True, dtype=torch.float64)
            # angle3 = torch.rand(1, requires_grad=True, dtype=torch.float64)

            optimizer = torch.optim.Rprop([params1, params2, angle1], lr=0.02)  # нашел минимум 1ф

            n_epochs = 80
            for epoch_index in range(n_epochs):
                optimizer.zero_grad()
                qp = forward(params1, params2, angle1[0])
                loss_value = loss(qp, params1, params2, angle1[0])
                loss_history.append(loss_value.cpu().data)
                loss_value.backward()
                optimizer.step()
            qp = forward(params1, params2, angle1[0]).cpu().data.item()
            if qp < best_qp:
                best_qp = qp
                # best_params1 = [params1.cpu().data, params2.cpu().data, angle1.cpu().data]
                best_params = [params1.cpu().data, params2.cpu().data, angle1.cpu().data]

            # print(qp)
            # print(loss_value)

            # plt.plot(loss_history, label='Loss')
            # plt.legend(loc='center right')
            # plt.show()

        print(f"{gamma:.1f}) ", sep='', end='', flush=True)
        print(best_qp, best_params, sep=', ', flush=True)
        qp_history.append(best_qp)
        params_history.append(best_params)

    # print(qp_history, flush=True)

    plt.plot(gammas, qp_history)
    plt.show()

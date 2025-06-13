import torch
from math import pi, sqrt, factorial, sin, cos, sinh, cosh
from torchquad import MonteCarlo, Gaussian, Trapezoid, Simpson, Boole
from wavefun import WaveFunStep, WaveFunContinuous


class Search:
    def __init__(self, device):
        """
        Search of the circuit using numeric methods.
        :param device: The device on which you want to store data and perform calculations (e.g. 'cuda').
        """
        self.device = device

    def __dot_product(self, n, x):
        return torch.exp(-x * x / 2) * torch.special.hermite_polynomial_h(x, n) / (pow(pi, 1 / 4) * pow(2, n / 2) * sqrt(factorial(n)))

    def __compute_output(self, input_fun1, input_fun2, n_measured_photons, bs_angle, output_density, integral_density):
        rho = cos(bs_angle)
        t = sin(bs_angle)

        # min_arg = -20
        # max_arg = 20
        min_arg = input_fun1.min_arg * rho + input_fun2.min_arg * t
        max_arg = input_fun1.max_arg * rho + input_fun2.max_arg * t
        x_2 = torch.arange(min_arg, max_arg, 1 / output_density, device=self.device)

        def fun(x):
            x = x.view(-1, 1)
            # return input_fun1(t * x + rho * x_2) * input_fun2(-rho * x + t * x_2)
            return input_fun1(t * x + rho * x_2) * input_fun2(-rho * x + t * x_2) * self.__dot_product(n_measured_photons, x)

        mc = Boole()
        # mc = MonteCarlo()
        result = mc.integrate(
            fun,
            dim=1,
            N=round(integral_density * (t * (input_fun1.max_arg - input_fun1.min_arg) + rho * (
                        input_fun2.max_arg - input_fun2.min_arg))),
            integration_domain=[
                [t * input_fun1.min_arg - rho * input_fun2.max_arg, t * input_fun1.max_arg - rho * input_fun2.min_arg]],
            # N=round(integral_density * 40),
            # integration_domain=[[-20, 20]],
            backend="torch",
        )

        return WaveFunStep(result, min_arg, max_arg)

    def __mean_O(self, input_fun, gamma, integral_density):
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

    def __mean_O_squared(self, input_fun, gamma, integral_density):
        fun_derivative = input_fun.derivative()
        fun_derivative2 = input_fun.derivative2()
        min_arg = min(fun_derivative.min_arg, fun_derivative2.min_arg)
        max_arg = max(fun_derivative.max_arg, fun_derivative2.max_arg)

        def fun(x):
            x = x.view(-1, 1)
            return input_fun(x).conj() * (-fun_derivative2(x) + gamma * x * (
                        (-2j + gamma * x * x * x) * input_fun(x) - 2j * x * fun_derivative(x)))

        mc = Boole()
        result = mc.integrate(
            fun,
            dim=1,
            N=round(integral_density * (max_arg - min_arg)),
            integration_domain=[[min_arg, max_arg]],
            backend="torch",
        )

        return result

    def __nonlinear_compression(self, input_fun, gamma, integral_density):
        return torch.real(self.__mean_O_squared(input_fun, gamma, integral_density) - self.__mean_O(input_fun, gamma,
                                                                                      integral_density) ** 2) * 2 / 3 * (
                    2 / gamma) ** (2 / 3)

    def forward(self, gamma, pattern, params1, params2, angle):
        # Инициализация функций
        A = 10
        r1 = params1[0]
        alpha1_real = params1[2]
        alpha1_imag = params1[3]
        fun1 = WaveFunContinuous(lambda x: torch.exp(
            1j * sqrt(2) * x * alpha1_imag - (x - sqrt(2) * alpha1_real) ** 2 / 2 / (
                    cosh(2 * r1) - sinh(2 * r1))) / pow(pi * (cosh(2 * r1) - sinh(2 * r1)), 1 / 4), -A / sqrt(2),
                                 A / sqrt(2))

        r2 = params2[0]
        phi = params2[1]
        alpha2_real = params2[2]
        alpha2_imag = params2[3]
        fun2 = WaveFunContinuous(lambda x: torch.exp(
            1j * sqrt(2) * x * alpha2_imag - (x - sqrt(2) * alpha2_real) ** 2 / 2 * (
                    1 + 1j * sin(phi) * sinh(2 * r2)) / (cosh(2 * r2) - cos(phi) * sinh(2 * r2))) / pow(
            pi * (cosh(2 * r2) - cos(phi) * sinh(2 * r2)), 1 / 4), -A / sqrt(2), A / sqrt(2))

        # Выходная функций, измерили паттерн
        fun = self.__compute_output(fun1, fun2, pattern[0], angle, 30, 15)

        # Нормализация
        vals = fun.values / fun.norm(15)
        gun = WaveFunStep(vals, -A, A)

        return self.__nonlinear_compression(gun, gamma, 15)

    def run(self, gamma, pattern, n_attempts, n_epochs=100):
        """
        Run the search.
        :param gamma: Parameter of cubic phase state.
        :param pattern: Pattern measured on the detectors.
        :param n_attempts: Number of attempts to find the local minimum.
        :param n_epochs: Number of epochs for every attempt.
        """
        best_nonlinear_compression = 1000
        best_params = []
        for attempt in range(n_attempts):
            # print("\r" + f"attempt {attempt}", end='')
            params1 = torch.rand(4, requires_grad=True, dtype=torch.float64)
            params2 = torch.rand(4, requires_grad=True, dtype=torch.float64)
            angle = torch.rand(1, requires_grad=True, dtype=torch.float64)

            optimizer = torch.optim.Adam([params1, params2, angle], lr=0.1, maximize=False)
            for epoch_index in range(n_epochs):
                optimizer.zero_grad()
                loss_value = self.forward(gamma, pattern, params1, params2, angle[0])
                loss_value.backward()
                optimizer.step()

            nonlinear_compression = self.forward(gamma, pattern, params1, params2, angle[0]).cpu().data.item()
            if nonlinear_compression < best_nonlinear_compression:
                best_nonlinear_compression = nonlinear_compression
                best_params = [params1.cpu().data, params2.cpu().data, angle.cpu().data]

        return best_nonlinear_compression, best_params

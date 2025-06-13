from math import log
from beamsplitter import BS
from special import hermite_H
from wavefun import WaveFunHermite
import torch


class Search:
    def __init__(self, n_modes, n_dimensions, device, precomputed_file):
        """
        Search of the circuit using series.
        :param n_modes: Number of modes in the circuit.
        :param n_dimensions: Number of first members of the series in the base decomposition.
        :param device: The device on which you want to store data and perform calculations (e.g. 'cuda').
        :param precomputed_file: Path to the precomputed tensor.
        """
        self.bs = BS(precomputed_file, device)
        self.n_modes = n_modes
        self.n_dimensions = n_dimensions

    def __bs_through(self, state, i, j, phi):
        """
        State after beam splitter.
        :param state: Initial state.
        :param i: First mode where beam splitter operates.
        :param j: Second mode where beam splitter operates.
        :param phi: Parameter of phase shifter.
        """
        state = state.transpose(i, -2)
        state = state.transpose(j, -1)

        shape = state.shape
        n = self.n_dimensions
        bs_transform = self.bs(phi).view(n, n, n)

        c = bs_transform.view(-1, n, n, n) * state.view(-1, 1, n, n)

        mask = torch.zeros(3*n - 2, n, dtype=torch.complex128)
        mask[n - 1:-(n - 1), :] = torch.eye(n).flip(1)

        c = c.reshape(-1, 1, n, n)
        mask = mask.view(1, 1, 3*n - 2, n)

        r = torch.nn.functional.conv2d(mask, c).view(-1, 2*n - 1).flip(1)

        index = torch.arange(n).view(1, -1) + (torch.arange(r.shape[0]) % n).view(-1, 1)

        result = torch.gather(r, 1, index).view(shape)

        result = result.transpose(j, -1)
        result = result.transpose(i, -2)
        return result

    def __ps_through(self, state, i, phi):
        """
        State ofter phase shifter.
        :param state: Input state.
        :param i: Number of mode where phase shifter operates.
        :param phi: Parameters of phase shifter.
        """
        n = self.n_dimensions
        m = self.n_modes

        shape = [1]*m
        shape[i] = n

        ps_transform = torch.exp(1j*phi)**torch.arange(n)
        ps_transform = ps_transform.view(*shape)

        return state * ps_transform

    def __mean_operator(self, fun, op):
        return fun.coeffs.conj() @ op @ fun.coeffs

    def __compute_ddx(self, n):
        """
        Derivative operator.
        :param n: Number of dimensions.
        """
        k = torch.arange(1, n)
        d = torch.sqrt(k/2)
        return torch.diag(d, 1) - torch.diag(d, -1)

    def __compute_x(self, n):
        """
        Operator of multiplication by x.
        :param n: Number of dimensions.
        """
        k = torch.arange(1, n)
        d = torch.sqrt(k/2)
        return torch.diag(d, 1) + torch.diag(d, -1)

    def __compute_O(self, gamma, n):
        """
        Operator O.
        :param gamma: Parameter of cubic phase
        :param n: Number of dimensions.
        """
        d = self.__compute_ddx(n)
        x = self.__compute_x(n)
        return -1j * d + gamma * (x @ x)

    def __nonlinear_compression(self, fun, gamma):
        # .to(torch.complex128)? Оптимизировать
        O = self.__compute_O(gamma, torch.numel(fun.coeffs)).to(torch.complex128)
        return (self.__mean_operator(fun, O @ O) - self.__mean_operator(fun, O)**2) / (1.5*(gamma/2)**(2/3))

    def __build_wave_fun(self, params, n):
        """
        Compute expansion coefficients of the function with respect to the Gauss-Hermite basis.
        :param params: Parameters that determine the function.
        :param n: Number of dimensions.
        """
        k = torch.arange(n).view(1, -1)

        r = params[..., 0].view(-1, 1)
        phi = params[..., 1].view(-1, 1)
        b1 = params[..., 2].view(-1, 1)
        b2 = params[..., 3].view(-1, 1)

        b = b1 + 1j*b2

        a = torch.exp(1j*phi)*torch.tanh(r)
        c = torch.exp(2*r)*torch.sin(phi/2)**2 + torch.exp(-2*r)*torch.cos(phi/2)**2

        return torch.sqrt(1 - a) * torch.exp((1 - a) * b**2 / 4 - b1**2 * c / 2 - 0.25*torch.log(c) - 0.5*(k*log(2) + torch.lgamma(k + 1)).squeeze()) * torch.diagonal(hermite_H(((1 - a)*b).view(-1), -a.view(-1), k)).t()

    def forward(self, gamma, pattern, topology, params, ps_angles, bs_angles):
        """
        Compute the circuit efficiency.
        :param gamma: Parameter of cubic phase state.
        :param pattern: Pattern measured on the detectors.
        :param topology: List of pairs of modes on which you want beam splitters to be.
        :param params: Parameters that determine the function.
        :param ps_angles: Angles of phase shifters.
        :param bs_angles: Angles of beam splitters.
        """
        # Инициализация
        coeffs = self.__build_wave_fun(params, self.n_dimensions)
        state = coeffs[0]
        for i in range(1, self.n_modes):
            state = torch.kron(state, coeffs[i])
        state = state.view(*([self.n_dimensions] * self.n_modes))

        # Для каждой пары мод применяем светоделитель, параметризуемый двумя числами
        counter = 0
        for pair in topology:
            state = self.__ps_through(state, pair[1], ps_angles[counter])
            state = self.__bs_through(state, pair[0], pair[1], bs_angles[counter])
            state = self.__ps_through(state, pair[1], -ps_angles[counter])

            counter += 1

        # Измерение
        coeffs = state[*pattern]

        # Нормализация
        coeffs = coeffs / torch.sqrt(coeffs.conj() @ coeffs)
        fun = WaveFunHermite(torch.cat((coeffs, torch.zeros(4)), 0))

        return torch.real(self.__nonlinear_compression(fun, gamma))

    def run(self, gamma, pattern, topology, n_attempts, n_epochs=100):
        """
        Run the search.
        :param gamma: Parameter of cubic phase state.
        :param pattern: Pattern measured on the detectors.
        :param topology: List of pairs of modes on which you want beam splitters to be.
        :param n_attempts: Number of attempts to find the local minimum.
        :param n_epochs: Number of epochs for every attempt.
        """
        best_nonlinear_compression = 1000
        best_params = []
        for attempt in range(n_attempts):
            # print("\r" + f"attempt {attempt}", end='')
            params = torch.rand(self.n_modes, 4, requires_grad=True, dtype=torch.float64)
            ps_angles = torch.rand((self.n_modes * (self.n_modes - 1)) // 2, requires_grad=True, dtype=torch.float64)
            bs_angles = torch.rand((self.n_modes * (self.n_modes - 1)) // 2, requires_grad=True, dtype=torch.float64)

            optimizer = torch.optim.Rprop([params, ps_angles, bs_angles], lr=0.02)

            for epoch_index in range(n_epochs):
                optimizer.zero_grad()
                loss_value = self.forward(gamma, pattern, topology, params, ps_angles, bs_angles)
                loss_value.backward()
                optimizer.step()

            nonlinear_compression = self.forward(gamma, pattern, topology, params, ps_angles, bs_angles).cpu().data.item()
            if nonlinear_compression < best_nonlinear_compression:
                best_nonlinear_compression = nonlinear_compression
                best_params = [params.cpu().data, ps_angles.cpu().data, bs_angles.cpu().data]

        return best_nonlinear_compression, best_params

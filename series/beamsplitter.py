import torch


class BS:
    """ Coefficients for computing state after beam splitter transform. """
    def __init__(self, filename, device):
        with open(filename, 'rb') as f:
            self.coeffs = torch.load(f, map_location=torch.device(device))
            self.n = (self.coeffs.shape[0] + 3) // 4

    def __call__(self, arg):
        n = torch.arange(2*self.n - 1)
        sin = torch.sin(n[1:].view(1, -1) * arg.view(-1, 1))
        cos = torch.cos(n.view(1, -1) * arg.view(-1, 1))
        basis = torch.cat((sin.flip(1), cos), 1)

        res = basis @ self.coeffs

        return res.view(-1, self.n, self.n, self.n)

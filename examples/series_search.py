import torch
import series
from topology import parallel_topology

device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    precomputed_file = '../precomputed/csc_tensor.pt'

    n_modes = 2
    n_dimensions = 50
    search = series.Search(n_modes, n_dimensions, device, precomputed_file)

    gamma = 0.1
    pattern = (1, )
    topology = parallel_topology(n_modes)

    n_attempts = 10

    print(search.run(gamma, pattern, topology, n_attempts))

import torch
import numeric
from torchquad import set_up_backend

import warnings
warnings.filterwarnings('ignore')

device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

if device == 'cuda':
    set_up_backend("torch", data_type="float64")

if __name__ == "__main__":
    search = numeric.Search(device)

    gamma = 0.1
    pattern = (1, )

    n_attempts = 10

    print(search.run(gamma, pattern, n_attempts))

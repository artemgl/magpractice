[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Производственная практика

# Пример поиска через разложение в ряды
Произведём поиск схемы с 2-мя каналами (`n_modes`) и одной измеренной частицей (`pattern`). Количество слагаемых в разложении функций по базису Гаусса-Эрмита (`n_dimensions`) равно 50. Параметр состояния кубической фазы (`gamma`) равен 0.1. Произведём поиск из 10-ти случайных точек (`n_attempts`).
```python
import torch
import series
from topology import parallel_topology

device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    precomputed_file = 'precomputed/csc_tensor.pt'

    n_modes = 2
    n_dimensions = 50
    search = series.Search(n_modes, n_dimensions, device, precomputed_file)

    gamma = 0.1
    pattern = (1, )
    topology = parallel_topology(n_modes)

    n_attempts = 10

    print(search.run(gamma, pattern, topology, n_attempts)[0])
```
Вывод:
```
0.7168215211921823
```
Алгоритм нашёл эффективную схему, генерирующую состояние с нелинейным сжатием около 0.72.

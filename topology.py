from math import ceil, log


def parallel_topology(n_modes):
    """ Parallel topology for the circuit. """
    res = []
    blocks = [n_modes]
    for i in range(ceil(log(n_modes, 2))):
        # Слой
        blocks = [x for sublist in [[b // 2, b - (b // 2)] for b in blocks] for x in sublist]
        block = []

        # Индекс, с которого начинаются преобразования в общей матрице
        start = 0
        for j in range(len(blocks) // 2):
            # Параллельный блок в слое

            # Количество мод в левой половине
            left = blocks[2 * j]
            # Количество мод в правой половине
            right = blocks[2 * j + 1]
            for k in range(right):
                # Параллельный шаг в блоке
                for m in range(left):
                    # Конкретная мода в левой половине
                    x = start + m
                    y = start + left + (m + k) % right
                    block.append([x, y])
            start += left + right
        res = block + res
    return res

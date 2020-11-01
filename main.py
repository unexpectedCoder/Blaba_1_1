from typing import Tuple
import numpy as np
import random
import os

import visualizer as vis


def main():
    fieldSize = 100                              # Размер поля клеток
    iters = 0                                    # Кол-во итераций "эволюции" клеточного автомата
    nAgents = 1000                               # Число агентов с каждой стороны
    datafile = 'data.npz'
    neighborhood = 'neumann'

    if os.path.isfile('data.npz'):
        choice = input("Выполнить моделирование (1) или показать только анимацию (2): ")

        if choice == '1':
            print("Инициализация...")
            cells = initialize(nAgents, fieldSize + 2)
            results = [cells[1:-1, 1:-1]]

            print("Моделирование...")
            if iters > 0:
                for _ in range(iters):
                    cells = updateCells(cells, neighborhood=neighborhood)
                    results.append(cells[1:-1, 1:-1])
                    if not np.any(cells[1:-1, 1:-1] == 1):
                        break
            else:
                while True:
                    cells = updateCells(cells, neighborhood=neighborhood)
                    results.append(cells[1:-1, 1:-1])
                    if not np.any(cells[1:-1, 1:-1] == 1):
                        break

            print("Сохранение результатов...")
            np.savez(datafile, *np.array(results))

            show(datafile)
        else:
            show(datafile)
    else:
        print("Инициализация...")
        cells = initialize(nAgents, fieldSize + 2)
        results = [cells[1:-1, 1:-1]]

        print("Моделирование...")
        if iters > 0:
            for _ in range(iters):
                cells = updateCells(cells, neighborhood=neighborhood)
                results.append(cells[1:-1, 1:-1])
                if not np.any(cells == 1):
                    break
        else:
            while True:
                cells = updateCells(cells, neighborhood='moore')
                results.append(cells[1:-1, 1:-1])
                if not np.any(cells[1:-1, 1:-1] == 1):
                    break

        print("Сохранение результатов...")
        np.savez(datafile, *np.array(results))

        show(datafile)

    return 0


def initialize(n_agents: int, field_size: int) -> np.ndarray:
    """Инициализирует начальное состояние клеточного автомата.

    :param n_agents: кол-во агентов с каждой стороны.
    :param field_size: размер матрицы (квадратной).
    :return: Инициализированный клеточный автомат (его начальное состояние).
    """
    inds = np.array([[i, j] for i in range(1, field_size-1) for j in range(1, field_size-1)])
    np.random.shuffle(inds)
    aind, bind = inds[:n_agents], inds[n_agents:2*n_agents]

    cells = np.ones((field_size, field_size))
    for a, b in zip(aind, bind):
        cells[a[0], a[1]] = 2   # A
        cells[b[0], b[1]] = 3   # B

    return cells


def updateCells(cells: np.ndarray, neighborhood: str = 'neumann') -> np.ndarray:
    """Реализует обновление (эволюцию) клеточного автомата в соответствие с правилами.

    :param cells: матрица клеток.
    :param neighborhood: тип окресности рассматриваемой клетки (фон Неймана или Мура).
    :return: Матрица с новыми значениями в клетках.
    """
    neighborhood = neighborhood.lower()
    newCells = cells.copy()

    for i in range(1, cells.shape[0] - 1):
        for j in range(1, cells.shape[1] - 1):
            na, nb = calcAB(cells, i, j, neighborhood)

            if isForcesEqual(na, nb):
                if newCells[i, j] == 1:
                    newCells[i, j] = 2 if random.choice((True, False)) else 3
                elif random.choice((True, False)):
                    newCells[i, j] = 1
            else:
                if na > nb and cells[i, j] == 3:
                    newCells[i, j] = 1
                elif na > nb and cells[i, j] == 1:
                    newCells[i, j] = 2
                elif na < nb and cells[i, j] == 2:
                    newCells[i, j] = 1
                elif na < nb and cells[i, j] == 1:
                    newCells[i, j] = 3
                else:
                    newCells[i, j] = cells[i, j]

    return newCells


def calcAB(cells: np.ndarray, i: int, j: int, neighborhood) -> Tuple[int, int]:
    """Подсчитать количество клеток со значением 2 и 3.

    :param cells: матрица клеток.
    :param i: текущая строка матрицы клеток.
    :param j: текущий столбец матрицы клеток.
    :param neighborhood: тип окресности рассматриваемой клетки (фон Неймана или Мура).
    :return: Количество клеток типа A и типа B.
    """
    if neighborhood == 'neumann':
        subcells = np.array([*cells[i-1:i+1, j], cells[i, j-1], cells[i, j+1]])
    elif neighborhood == 'moore':
        subcells = cells[i-1:i+2, j-1:j+2].ravel()
    else:
        subcells = None
    return int(np.sum(subcells == 2)), \
           int(np.sum(subcells == 3))


def isForcesEqual(na: int, nb: int) -> bool:
    if na != 0 and na == nb:
        return True
    return False


def show(datafile: str):
    data = np.load(datafile)
    dataList = [d for d in data.values()]

    v = vis.Visualizer()
    v.show(dataList, len(dataList))


if __name__ == '__main__':
    main()

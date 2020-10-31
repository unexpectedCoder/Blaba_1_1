from typing import Tuple
import numpy as np
import random
import os

import visualizer as vis


def main():
    n = 150                                     # Размер поля клеток
    iters = 200                                 # Кол-во итераций "эволюции" клеточного автомата
    cells = initialize(n+2, variant=1)          # Поле клеток
    datafile = 'data.npz'

    if os.path.isfile('data.npz'):
        choice = input("Выполнить моделирование (1) или показать только анимацию (2): ")

        if choice == '1':
            results = []

            for _ in range(iters):
                cells = updateCells(cells, neighborhood='moore')
                results.append(cells[1:-2, 1:-2])  # Не записываются добавленные граничные строки и столбцы

            np.savez(datafile, *np.array(results))
            show(datafile)
        else:
            show(datafile)
    else:
        results = []
        for _ in range(iters):
            cells = updateCells(cells, neighborhood='moore')
            results.append(cells[1:-2, 1:-2])  # Не записываются добавленные граничные строки и столбцы

        np.savez(datafile, *np.array(results))
        show(datafile)

    return 0


def initialize(n: int, variant: int = 1) -> np.ndarray:
    """Инициализирует начальное состояние клеточного автомата.

    :param n: размер матрицы (квадратной).
    :param variant: номер варианта начального состояния.
    :return: Инициализированный клеточный автомат (его начальное состояние).
    """
    if variant == 1:
        cells = np.random.randint(1, 4, (n, n))
        return cells
    if variant == 2:
        cells = np.random.randint(1, 4, (n, n))
        cells[n//2 - n//4: n//2 + n//4 + 1, n//2 - n//4: n//2 + n//4 + 1] = 1
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
            if na + nb > 0:
                pa, pb = winWithProbability(na / (na + nb)), winWithProbability(nb / (na + nb))
            else:
                pa, pb = False, False

            if na >= nb and cells[i, j] == 3:
                if pa:
                    newCells[i, j] = 1
            elif na > nb and cells[i, j] == 1:
                if pa:
                    newCells[i, j] = 2
            elif na <= nb and cells[i, j] == 2:
                if pb:
                    newCells[i, j] = 1
            elif na < nb and cells[i, j] == 1:
                if pb:
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


def winWithProbability(p: float) -> bool:
    if 0 <= p <= 1:
        return random.random() <= p
    return True


def show(datafile: str):
    data = np.load(datafile)
    dataList = [d for d in data.values()]

    v = vis.Visualizer()
    v.show(dataList, len(dataList))


if __name__ == '__main__':
    main()

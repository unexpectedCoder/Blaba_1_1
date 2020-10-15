from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def main():
    n = 150                                     # Размер поля клеток
    iters = 75                                  # Кол-во итераций "эволюции" клеточного автомата
    cells = initialize(n+2, variant=2)          # Поле клеток

    # Обновление (эволюция) состояния клеточного автомата
    results = []
    for _ in range(iters):
        cells = updateCells(cells, neighborhood='neumann')
        results.append(cells[1:-2, 1:-2])       # Не записываются добавленные граничные строки и столбцы

    # Запись накопленных данных
    datafile = 'data.npz'
    np.savez(datafile, *np.array(results))

    # Вывод с анимацией
    show(datafile, iters)

    return 0


def initialize(n: int, variant: int = 1) -> np.ndarray:
    """Инициализирует начальное состояние клеточного автомата.

    :param n: размер матрицы (квадратной).
    :param variant: номер варианта начального состояния.
    :return: Инициализированный клеточный автомат (его начальное состояние).
    """
    if variant == 1:
        cells = np.random.randint(1, 4, (n, n))
        cells[0, :] = 1
        cells[-1, :] = 1
        cells[:, 0] = 1
        cells[:, -1] = 1
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
    buf = cells.copy()

    for i in range(1, cells.shape[0] - 1):
        for j in range(1, cells.shape[1] - 1):
            na, nb = calcAB(cells, i, j, neighborhood)
            if na > nb:
                buf[i, j] = 2
            elif na < nb:
                buf[i, j] = 3
            else:
                buf[i, j] = cells[i, j]

    return buf


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
    elif neighborhood == 'mur':
        subcells = cells[i-1:i+2, j-1:j+2].ravel()
    else:
        subcells = None
    return len([x for x in subcells if x == 2]), \
           len([x for x in subcells if x == 3])


def show(datafile, iters: int):
    """Показать анимацию эволюции клеточного автомата.

    :param datafile: имя файла с данными numpy.
    :param iters: количество проведенных итераций эволюции клеточного автомата (кол-во матриц в файле).
    """
    data = np.load(datafile)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ani = animation.FuncAnimation(fig, animate, fargs=(ax, data, iters), interval=250)
    plt.show()


def animate(i, ax, data, iters: int):
    """Функция анимации."""
    if i < iters:
        ax.clear()
        d = data[f'arr_{i}']        # arr_0, arr_1 и т.д. - это ключи по-умолчанию для numpy.savez()
        ax.matshow(d, cmap='jet')
    else:
        print("Усё")


if __name__ == '__main__':
    main()

from typing import List, Tuple
from pygame.locals import *
import pygame
import numpy as np


WHITE = 255, 255, 255
ORANGE = 255, 102, 0
GREEN = 102, 255, 51
GREY = 150, 150, 150
BLACK = 0, 0, 0


class Visualizer:
    def __init__(self, win_size: Tuple[int, int] = (900, 900)):
        self._winSize = win_size
        self.lx, self.ly = None, None
        self.xmargin, self.ymargin = None, None

        self._fps = 25
        self.fpsClock = pygame.time.Clock()

        # PyGame
        pygame.init()
        self.DISPLAYSURF = pygame.display.set_mode(self._winSize)
        pygame.display.set_caption("Cellular Automata")

    def show(self, data: List[np.ndarray], iters: int):
        fontObj = pygame.font.Font('roboto.ttf', 20)

        i = 0
        isRun, start = True, False
        while True:
            self.fpsClock.tick(self._fps)
            self.drawField(data[i])

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    isRun = False
                if event.type == KEYDOWN:
                    if event.key == K_RIGHT:
                        i += 1 if i < iters - 1 else 0
                    if event.key == K_LEFT:
                        i -= 1 if i > 0 else 0
                    if event.key == K_UP:
                        self._fps += 1 if self._fps < 30 else 0
                    if event.key == K_DOWN:
                        self._fps -= 1 if self._fps > 1 else 0
                    if event.key == K_b:
                        i = 0
                    if event.key == K_e:
                        i = iters - 1
                    if event.key == K_SPACE and not start:
                        start = True if i < iters - 1 else False
                    elif start:
                        start = False
            if not isRun:
                break

            txt = fontObj.render(f'Итерация #{i}   FPS: {self.fpsClock.get_fps():.1f}', True, BLACK)
            self.DISPLAYSURF.blit(txt, (0, 0))
            pygame.display.update()

            if start:
                i += 1
                if i == iters - 1:
                    start = False

    def drawField(self, data: np.ndarray):
        self.DISPLAYSURF.fill(WHITE)

        self.lx = int(self._winSize[0] / data.shape[1])
        self.ly = int(self._winSize[1] / data.shape[0])
        self.xmargin = int(self._winSize[0] - data.shape[1] * self.lx) // 2
        self.ymargin = int(self._winSize[1] - data.shape[0] * self.ly) // 2

        rects = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                width = 0
                if data[i, j] == 2:
                    color = ORANGE
                elif data[i, j] == 3:
                    color = GREEN
                else:
                    color = GREY
                    width = 1
                rects.append(((j * self.lx + self.xmargin, i * self.ly + self.ymargin, self.lx, self.ly), color, width))

        for rect in rects:
            pygame.draw.rect(self.DISPLAYSURF, rect[1], rect[0], rect[2])

import os
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from tools.Pixel import Pixel


class PixelArray(ABC):

    @property
    @abstractmethod
    def n_dimensions(self):
        pass


class Image2D(PixelArray):

    def __init__(self, filename, kind='rgb'):
        if not os.path.exists(filename):
            raise FileNotFoundError
        self.filename = os.path.abspath(filename)
        self.__im = None

    @property
    def n_dimensions(self):
        return 2

    @property
    def im(self):
        if self.__im is None:
            self.__im = np.array([[Pixel.create(p) for p in row] for row in np.array(Image.open(self.filename))])
        return self.__im

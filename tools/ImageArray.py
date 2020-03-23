import os

import numpy as np
from PIL import Image


class ImageArray:
    __slots__ = ['__im', '__head', '__tail']

    def __init__(self, filename):
        self.__im = None
        self.__head, self.__tail = os.path.split(filename)
        self.reload()

    def reload(self):
        self.__im = np.array(Image.open(self.filename).convert('RGBA'))

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        Image.fromarray(self.im[::3]).save(filename)

    @property
    def filename(self):
        return os.path.join(self.__head, self.__tail)

    @property
    def im(self):
        return self.__im

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return self.im.shape[:2]

    @property
    def w(self):
        return self.shape[0]

    @property
    def h(self):
        return self.shape[1]

    @property
    def R(self):
        return self.im[:, :, 0]

    @R.setter
    def R(self, r):
        self.im[:, :, 0] = r

    @property
    def G(self):
        return self.im[:, :, 1]

    @G.setter
    def G(self, g):
        self.im[:, :, 1] = g

    @property
    def B(self):
        return self.im[:, :, 2]

    @B.setter
    def B(self, b):
        self.im[:, :, 2] = b

    @property
    def A(self):
        return self.im[:, :, 3]

    @A.setter
    def A(self, a):
        self.im[:, :, 3] = a

    @property
    def H(self):
        h = np.zeros(self.shape)
        c_max = self.V
        delta = c_max - np.min((self.R, self.G, self.B), axis=0) / 255
        idx = np.logical_and(delta != 0, c_max == self.R / 255)
        h[idx] = (self.G[idx] - self.B[idx]) / delta[idx] % 6
        idx = np.logical_and(delta != 0, c_max == self.G / 255)
        h[idx] = (self.B[idx] - self.R[idx]) / delta[idx] + 2
        idx = np.logical_and(delta != 0, c_max == self.B / 255)
        h[idx] = (self.R[idx] - self.G[idx]) / delta[idx] + 4
        return 60 * h

    @H.setter
    def H(self, h):
        raise NotImplementedError

    @property
    def S(self):
        s = np.zeros(self.shape)
        c_max = self.V
        delta = c_max - np.min((self.R, self.G, self.B), axis=0) / 255
        idx = c_max != 0
        s[idx] = delta[idx] / c_max[idx]
        return s

    @S.setter
    def S(self, s):
        raise NotImplementedError

    @property
    def L(self):
        return (self.V + np.min((self.R, self.G, self.B), axis=0) / 255) / 2

    @L.setter
    def L(self, l):
        raise NotImplementedError

    @property
    def V(self):
        return np.max((self.R, self.G, self.B), axis=0) / 255

    @V.setter
    def V(self, v):
        raise NotImplementedError

    @property
    def C(self):
        return 1 - (self.R / 255) / (1 - self.K)

    @C.setter
    def C(self, c):
        raise NotImplementedError

    @property
    def M(self):
        return 1 - (self.G / 255) / (1 - self.K)

    @M.setter
    def M(self, m):
        raise NotImplementedError

    @property
    def Y(self):
        return 1 - (self.B / 255) / (1 - self.K)

    @Y.setter
    def Y(self, y):
        raise NotImplementedError

    @property
    def K(self):
        return 1 - np.max((self.R, self.G, self.B), axis=0) / 255

    @K.setter
    def K(self, k):
        raise NotImplementedError


if __name__ == '__main__':
    im = ImageArray('../originals/gandhi_1.jpeg')

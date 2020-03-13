from abc import ABC, abstractmethod

import numpy as np


class Pixel(ABC):
    """
    Abstract class to represent a pixel.
    """

    @staticmethod
    def create(arg=(0, 0, 0, 1), kind: str = 'rgb'):
        if len(arg) == 3:
            return Pixel.create((arg[0], arg[1], arg[2], 1), kind)
        elif len(arg) != 4:
            raise ValueError
        elif kind.lower().startswith('rgb'):
            return RGB(arg)
        elif kind.lower().startswith('hsl'):
            raise NotImplementedError  # TODO
        elif kind.lower().startswith('hsv'):
            raise NotImplementedError  # TODO
        elif kind.lower().startswith('cmyk'):
            raise NotImplementedError  # TODO

    @property
    @abstractmethod
    def vector(self):
        pass

    @property
    @abstractmethod
    def rgba(self):
        pass

    @property
    @abstractmethod
    def hslva(self):
        pass

    @property
    @abstractmethod
    def hsla(self):
        pass

    @property
    @abstractmethod
    def hsva(self):
        pass

    @property
    @abstractmethod
    def cmyk(self):
        pass

    @property
    @abstractmethod
    def a(self):
        pass

    @property
    def rgb(self):
        return self.rgba[:3]

    @property
    def hslv(self):
        return self.hslva[:4]

    @property
    def hsl(self):
        return self.hsla[:3]

    @property
    def hsv(self):
        return self.hsva[:3]

    @property
    def r(self):
        return self.rgba[0]

    @property
    def g(self):
        return self.rgba[1]

    @property
    def b(self):
        return self.rgba[2]

    @property
    def h(self):
        return self.hslva[0]

    @property
    def s(self):
        return self.hslva[1]

    @property
    def l(self):
        return self.hslva[2]

    @property
    def v(self):
        return self.hslva[3]

    @property
    def c(self):
        return self.cmyk[0]

    @property
    def m(self):
        return self.cmyk[1]

    @property
    def y(self):
        return self.cmyk[2]

    @property
    def k(self):
        return self.cmyk[3]

    @staticmethod
    def to_int_0_255(x):
        if int(x) == x:
            return int(x)
        elif 0 < x < 1:
            return int(x * 256)
        else:
            raise ValueError

    @staticmethod
    def to_float_0_1(x):
        if 0 <= x <= 1:
            return float(x)
        elif int(x) == x and 0 <= x < 256:
            return float(x) / 255
        else:
            raise ValueError

    def __repr__(self):
        return repr(list(self.vector))


class RGB(Pixel):
    """
    Class to represent a RGB(A) pixel.
    """

    __slots__ = ['__rgba']

    def __init__(self, vector=(0, 0, 0, 1)):
        self.__rgba = np.array(vector)

    @property
    def vector(self):
        return self.rgba

    @property
    def rgba(self):
        return self.__rgba

    @property
    def hslva(self):
        _r, _g, _b = self.rgb / 255
        c_min = min(_r, _g, _b)
        c_max = max(_r, _g, _b)
        delta = c_max - c_min
        l = (c_max + c_min) / 2
        if delta == 0:
            h, s = 0, 0
        else:
            s = delta / (1 - abs(2 * l - 1))
            if c_max == _r:
                h = 60 * (((_g - _b) / delta) % 6)
            elif c_max == _g:
                h = 60 * (((_b - _r) / delta) + 2)
            else:
                h = 60 * (((_r - _g) / delta) + 4)
        v = l * s * min(l, 1 - l)
        return np.array([h, s, l, v, self.a])

    @property
    def hsla(self):
        return np.array([self.h, self.s, self.l, self.a])

    @property
    def hsva(self):
        return np.array([self.h, self.s, self.v, self.a])

    @property
    def cmyk(self):
        return RGB.cmyk_from_rgba(self.rgba)

    @property
    def a(self):
        return self.rgba[3]

    @staticmethod
    def cmyk_from_rgba(rgba):
        cmyk = rgba / 255
        cmyk[3] = np.max(cmyk[:3])
        cmyk[:3] = 1 - cmyk[:3] / cmyk[3]
        cmyk[3] = 1 - cmyk[3]
        return cmyk


class HSLV(Pixel, ABC):
    """
    Abstract class to encapsulate shared behaviour of HSL and HSV.
    """

    def to_rgba(self, c):
        x = c * (1 - abs((self.h / 60) % 2 - 1))
        m = self.l - c / 2
        if self.h < 60:
            rgba = np.array([c, x, 0, 0])
        elif self.h < 120:
            rgba = np.array([x, c, 0, 0])
        elif self.h < 180:
            rgba = np.array([0, c, x, 0])
        elif self.h < 240:
            rgba = np.array([0, x, c, 0])
        elif self.h < 300:
            rgba = np.array([x, 0, c, 0])
        else:
            rgba = np.array([c, 0, x, 0])
        rgba += m
        rgba *= 255
        rgba[3] = self.a
        return rgba

    @property
    def cmyk(self):
        return RGB.cmyk_from_rgba(self.rgba)


class HSL(HSLV):
    """
    Class to represent a HSL(A) pixel.
    """

    __slots__ = ['__hsla']

    def __init__(self, vector=(0, 0, 0, 1)):
        self.__hsla = np.array(vector)

    @property
    def vector(self):
        return self.hsla

    @property
    def rgba(self):
        return self.to_rgba(self.s * (1 - abs(2 * self.l - 1)))

    @property
    def hslva(self):
        h, s, l, a = self.hsla
        return np.array([h, s, l, self.v, a])

    @property
    def hsla(self):
        return self.__hsla

    @property
    def hsva(self):
        h, s, _, a = self.hsla
        return np.array([h, s, self.v, a])

    @property
    def v(self):
        return self.l * self.s * min(self.l, 1 - self.l)

    @property
    def a(self):
        return self.hsla[3]


class HSV(HSLV):
    """
    Class to represent a HSV(A) pixel.
    """

    __slots__ = ['__hsva']

    def __init__(self, vector=(0, 0, 0, 1)):
        self.__hsva = np.array(vector)

    @property
    def vector(self):
        return self.hsva

    @property
    def rgba(self):
        return self.to_rgba(self.s * self.v)

    @property
    def hslva(self):
        h, s, v, a = self.hsva
        return np.array([h, s, self.l, v, a])

    @property
    def hsla(self):
        h, s, _, a = self.hsva
        return np.array([h, s, self.l, a])

    @property
    def hsva(self):
        return self.__hsva

    @property
    def l(self):
        return self.v * (1 - self.s / 2)

    @property
    def a(self):
        return self.hsva[3]

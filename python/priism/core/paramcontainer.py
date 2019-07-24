# Copyright (C) 2019
# National Astronomical Observatory of Japan
# 2-21-1, Osawa, Mitaka, Tokyo, 181-8588, Japan.
#
# This file is part of PRIISM.
#
# PRIISM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# PRIISM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PRIISM.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import absolute_import


class ParamContainer(object):
    def InitContainer(self, kwargs):
        ignores = ('self',)
        for k, v in kwargs.items():
            if k not in ignores:
                setattr(self, k, v)

    @classmethod
    def CreateContainer(container_cls, **kwargs):
        ignores = ('self',)
        kw = kwargs.copy()
        for key in ignores:
            if key in kw:
                kw.pop(key)
        return container_cls(**kw)


class MfistaParamContainer(ParamContainer):
    def __init__(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None):
        """
        Constructor
        """
        self.InitContainer(locals())


class SimpleImageParamContainer(ParamContainer):
    """
    This is primitive image parameter container that
    specifies three-dimensional cube (imsize for spatial axes
    and nchan for spectral axis).
    """
    def __init__(self, imsize=100, nchan=-1):
        self.InitContainer(locals())

    @property
    def imsize(self):
        return getattr(self, '_imsize', None)

    @imsize.setter
    def imsize(self, value):
        if hasattr(value, '__iter__'):
            self._imsize = list(value)
        else:
            self._imsize = [int(value)]

        if len(self._imsize) == 0:
            raise TypeError('given imsize is not correct')
        elif len(self._imsize) == 1:
            self._imsize = [self._imsize[0], self._imsize[0]]
        else:
            self._imsize = self._imsize[:2]

# Copyright (C) 2019-2022
# Inter-University Research Institute Corporation, National Institutes of Natural Sciences
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
from __future__ import print_function

import math
import numpy as np

import priism.external.sakura as sakura
from . import paramcontainer


def exec_line(f, varname):
    line = f.readline()
    exec(line.rstrip('\n'))
    val = locals()[varname]
    #print '{0} = {1}'.format(varname, val)
    return val


class GriddedVisibilityStorage(object):
    """
    Class to hold gridder result

    expected array shape for grid_real and grid_imag is
    (nv, nu, npol, nchan)
    """
    @classmethod
    def importdata(cls, filename):
        with open(filename, 'r') as f:
            # read M
            M = exec_line(f, 'M')

            # read NX
            NX = exec_line(f, 'NX')

            # read NY
            NY = exec_line(f, 'NY')

            # skip headers
            f.readline()
            f.readline()
            f.readline()

            # read input data
            grid_shape = (NY, NX, 1, 1,)
            yreal = np.zeros(grid_shape, dtype=np.float64)
            yimag = np.zeros_like(yreal)
            weight = np.zeros_like(yreal)
            #u = np.empty(M, dtype=np.int32)
            #v = np.empty_like(u)
            #yreal = np.empty(M, dtype=np.double)
            #yimag = np.empty_like(yreal)
            #noise = np.empty_like(yreal)
            for i in range(M):
                line = f.readline()
                values = line.split(',')
                u = np.int32(values[0].strip())
                v = np.int32(values[1].strip())
                yreal[v, u, 0, 0] = np.double(values[2].strip())
                yimag[v, u, 0, 0] = np.double(values[3].strip())
                noise = np.double(values[4].strip())
                weight[v, u, 0, 0] = 1 / (noise * noise)
                #print '{0} {1} {2} {3}'.format(u[i], v[i], yreal[i], yimag[i], noise[i])

            storage = cls(yreal, yimag, weight)
            return storage

    def __init__(self, grid_real, grid_imag, wgrid_real, wgrid_imag=None, num_ws=None):
        self.real = grid_real
        self.imag = grid_imag
        self.wreal = wgrid_real
        self.wimag = wgrid_imag if wgrid_imag is not None else wgrid_real
        assert self.real.shape == self.imag.shape
        assert self.real.shape == self.wreal.shape
        assert self.real.shape == self.wimag.shape
        self.shape = self.real.shape

        # number of ws that are accumulated onto grid
        self.num_ws = num_ws if num_ws is not None else 0

    def exportdata(self, filename):
        nonzeros = np.where(self.wreal != 0.0)
        m = len(nonzeros[0])
        nx = self.shape[1]
        ny = self.shape[0]
        with open(filename, 'w') as f:
            print('M = {0}'.format(m), file=f)
            print('NX = {0}'.format(nx), file=f)
            print('NY = {0}'.format(ny), file=f)
            print('', file=f)
            print('u, v, y_r, y_i, noise_std_dev', file=f)
            print('', file=f)
            for i in range(m):
                u = nonzeros[1][i]
                v = nonzeros[0][i]
                noise = 1 / math.sqrt(self.wreal[v, u, 0, 0])
                print('{0}, {1}, {2:e}, {3:e}, {4:e}'.format(u,
                                                             v,
                                                             self.real[v, u, 0, 0],
                                                             self.imag[v, u, 0, 0],
                                                             noise), file=f)


class ResultingImageStorage(object):
    """
    Class to hold imaging result

    expected array shape for the image is (nx, ny, npol, nchan)
    """
    def __init__(self, data):
        self.data = data


class UVGridConfig(paramcontainer.ParamContainer):
    """
    Class to hold grid configuration on u-v plane
    """
    @property
    def offsetu(self):
        return self._offsetu if self._offsetu is not None else self.nu // 2

    @offsetu.setter
    def offsetu(self, value):
        self._offsetu = value

    @property
    def offsetv(self):
        return self._offsetv if self._offsetv is not None else self.nv // 2

    @offsetv.setter
    def offsetv(self, value):
        self._offsetv = value

    def __init__(self, cellu, cellv, nu, nv, offsetu=None, offsetv=None):
        self.InitContainer(locals())


class VisibilityWorkingSet(paramcontainer.ParamContainer):
    """
    Working set for visibility data

    NOTE: flag=True indicates *VALID* data
          flag=False indicates *INVALID* data

    data_id --- arbitrary data ID
    u, v --- position in uv-plane as pixel coordinate (nrow)
    rdata --- real part of visibility data (nrow, npol, nchan)
    idata --- imaginary part of visibility data (nrow, npol, nchan)
    weight --- visibility weight (nrow, nchan)
    """
    def __init__(self, data_id=None, u=0.0, v=0.0, rdata=None, idata=None,
                 weight=None):
        self.InitContainer(locals())

    def __len__(self):
        return len(self.u)

    def __from_shape(self, axis=0):
        if self.rdata is None:
            return 0
        else:
            # should be numpy array
            shape = self.rdata.shape
            if len(shape) < axis + 1:
                return 1
            else:
                return shape[axis]

    @property
    def nrow(self):
        return self.__from_shape(axis=0)

    @property
    def nchan(self):
        return self.__from_shape(axis=2)

    @property
    def npol(self):
        return self.__from_shape(axis=1)

    @property
    def start(self):
        return 0

    @property
    def end(self):
        return self.nrow - 1

    @property
    def data_id(self):
        if hasattr(self, '_data_id'):
            return self._data_id
        else:
            raise ValueError('invalid data_id: Undefined')

    @data_id.setter
    def data_id(self, value):
        if not isinstance(value, int):
            raise ValueError('invalid data_id ({0}). Should be int'.format(value))
        else:
            self._data_id = value


def grid2ws(grid_real, grid_imag, wgrid_real, wgrid_imag):
    """convert gridder result into workingset instance

    Arguments:
        grid_real {np.ndarray} -- real part of the visibility
        grid_imag {np.ndarray} -- imaginary part of the visibility
        wgrid_real {np.ndarray} -- weight of the visibility
        wgrid_imag {np.ndarray} -- weight of the visibility

    Returns:
        VisibilityWorkingSet -- visibility working set
    """
    gridshape = grid_real.shape
    nonzero_real = np.where(wgrid_real != 0.0)
    nonzero_imag = np.where(wgrid_imag != 0.0)
    # uv location
    # assumption here is that the first index corresponds to v while
    # the second one corresponds u so that [i,j] represents
    # the value at uv location (j,i).
    # since the array is C-contiguous, memory layout is contiguous
    # along u-axis.
    #
    # | 9|10|11|
    # | 6| 7| 8|
    # | 3| 4| 5|
    # | 0| 1| 2|
    npol = gridshape[2]
    nchan = gridshape[3]
    data_id = 0
    num_vis = len(nonzero_real[0])
    u = sakura.empty_aligned((num_vis,), dtype=np.int32)
    v = sakura.empty_like_aligned(u)
    rdata = sakura.empty_aligned((num_vis,), dtype=np.float64)
    idata = sakura.empty_like_aligned(rdata)
    wdata = sakura.empty_like_aligned(rdata)
    xpos = 0
    for ipol in range(npol):
        for ichan in range(nchan):
            ir = np.where(np.logical_and(nonzero_real[2] == ipol,
                                               nonzero_real[3] == ichan))
            #ii = np.where(np.logical_and(nonzero_imag[2] == ipol,
            #                                   nonzero_imag[3] == ichan))
            xlen = len(v)
            nextpos = xpos + xlen
            v[xpos:nextpos] = nonzero_real[0][ir]
            u[xpos:nextpos] = nonzero_real[1][ir]
            rdata[xpos:nextpos] = grid_real[v, u, ipol, ichan]
            idata[xpos:nextpos] = grid_imag[v, u, ipol, ichan]
            wdata[xpos:nextpos] = wgrid_real[v, u, ipol, ichan]
            xpos = nextpos
    visibility_data = VisibilityWorkingSet(data_id=data_id,
                                           u=u,
                                           v=v,
                                           rdata=rdata,
                                           idata=idata,
                                           weight=wdata)
    return visibility_data

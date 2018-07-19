from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import math
import numpy
import collections
import pylab as pl
import matplotlib
import time

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
            yreal = numpy.zeros(grid_shape, dtype=numpy.float64)
            yimag = numpy.zeros_like(yreal)
            weight = numpy.zeros_like(yreal)
            #u = numpy.empty(M, dtype=numpy.int32)
            #v = numpy.empty_like(u)
            #yreal = numpy.empty(M, dtype=numpy.double)
            #yimag = numpy.empty_like(yreal)
            #noise = numpy.empty_like(yreal)
            for i in range(M):
                line = f.readline()
                values = line.split(',')
                u = numpy.int32(values[0].strip())
                v = numpy.int32(values[1].strip())
                yreal[v, u, 0, 0] = numpy.double(values[2].strip())
                yimag[v, u, 0, 0] = numpy.double(values[3].strip())
                noise = numpy.double(values[4].strip())
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
        nonzeros = numpy.where(self.wreal != 0.0)
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
    
    u, v --- position in uv-plane (nrow)
    rdata --- real part of visibility data (nrow, npol, nchan)
    idata --- imaginary part of visibility data (nrow, npol, nchan)
    flag --- channelized flag (nrow, npol, nchan)
    weight --- visibility weight (nrow, nchan) 
    row_flag --- row flag (nrow)
    channel_map --- channel mapping between raw visibility
                    and gridded visibility (nchan)
    pol_map --- polarization mapping between raw visibility
                and gridded visibility (npol)
    """
    def __init__(self, data_id=None, u=0.0, v=0.0, rdata=None, idata=None, 
                 weight=None):
        self.InitContainer(locals())
        
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
            
#     @property
#     def pol_map(self):
#         if hasattr(self, '_pol_map'):
#             return self._pol_map
#         else:
#             self._pol_map = sakura.empty_aligned((self.npol,), dtype=numpy.int32)
#             self._pol_map[:] = numpy.arange(self.npol)
#             return self._pol_map
#     
#     @pol_map.setter
#     def pol_map(self, value):
#         if value is None:
#             #self._pol_map = sakura.empty_aligned((self.npol,), dtype=numpy.int32)
#             #self._pol_map[:] = range(self.npol)
#             pass
#         else:
#             try:
#                 if len(value) == self.npol and isinstance(value[0], int):
#                     self._pol_map = value
#                 else:
#                     raise
#             except Exception, e:
#                 raise ValueError('invalid pol_map ({0}). Should be int list or None.'.format(value))
    
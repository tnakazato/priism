from __future__ import absolute_import

import os
import shutil
import math
import numpy
import collections
import pylab as pl
import matplotlib
import time

from . import paramcontainer

class GriddedVisibilityStorage(object):
    """
    Class to hold gridder result
    
    expected array shape for grid_real and grid_imag is 
    (nv, nu, npol, nchan)
    """
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
    def __init__(self, cellu, cellv, nu, nv, offsetu, offsetv):
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
    
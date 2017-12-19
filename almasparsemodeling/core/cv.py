from __future__ import absolute_import

import numpy
import scipy
import itertools

import almasparsemodeling.external.sakura as sakura
from . import util


class GriddedVisibilitySubsetHandler(object):
    def __init__(self, griddedvis, uvgridconfig, num_fold=10):
        self.visibility = griddedvis
        self.uvgrid = uvgridconfig
        self.num_fold = num_fold
        
        self._clear
        
    def _clear(self):
        self.visibility_active = None
        self.visibility_cache = None
        self.active_index = None
        self.subset_id = None
        
        # grid data shape: (nv, nu, npol, nchan)
        grid_real = self.visibility.real
        grid_imag = self.visibility.imag
        
        # amplitude should be nonzero in active pixels
        self.active_index = numpy.where(numpy.logical_and(grid_real != 0, grid_imag != 0))
        num_active = len(self.active_index[0])
    
        # random index 
        self.index_generator = util.VisibilitySubsetGenerator(num_active, self.num_fold)
      
    def generate_subset(self, subset_id):
        self.subset_id = subset_id
        
        # grid data shape: (nv, nu, npol, nchan)
        grid_real = self.visibility.real
        grid_imag = self.visibility.imag
        gdata_shape = grid_real.shape
        
        # random index 
        random_index = self.index_generator.get_subset_index(self.subset_id)
        
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
        u = sakura.empty_aligned((num_subvis), dtype=numpy.float64)
        v = sakura.empty_like_aligned(u)
        uid = self.active_index[1][random_index]
        vid = self.active_index[0][random_index]
        cellu = self.uvgrid.cellu
        cellv = self.uvgrid.cellv
        offsetu = self.uvgrid.offsetu
        offsetv = self.uvgrid.offsetv
        nu = self.uvgrid.nu
        nv = self.uvgrid.nv
        assert gdata_shape[0] == nv
        assert gdata_shape[1] == nu
        u[:] = (uid % nu - offsetu) * cellu
        v[:] = (vid / nu - offsetv) * cellv
        
        # visibility data to be cached
        # here, we assume npol == 1 (Stokes visibility I_v) and nchan == 1
        assert len(gdata_shape) == 4
        assert gdata_shape[2] == 1 # npol should be 1
        assert gdata_shape[3] == 1 # nchan should be 1 
        data_shape = (num_subvis)
        real = sakura.empty_like_aligned(data_shape, dtype=numpy.float32)
        imag = sakura.empty_like_aligned(real)
        
        real[:] = grid_real[self.active_index][random_index]
        imag[:] = grid_imag[self.active_index][random_index]
        
        # generate subset
        self.visibility_active = self.visibility
        self.__replace_with(self.visibility_active.real, random_index, 0.0)
        self.__replace_with(self.visibility_active.imag, random_index, 0.0)
        self.visibility_cache = [gridder.GridderWorkingSet(rdata=real, 
                                                           idata=imag,
                                                           u=u,
                                                           v=v)]
        
    def restore_visibility(self):
        if self.visibility_active is not None and self.visibility_cache is not None:
            self.__replace_with(self.visibility.real, self.visibility_cache.rdata)
            self.__replace_with(self.visibility.imag, self.visibility_cache.idata)
            self._clear()
            
    def __replace_with(self, src, index_list, newval):
        src[self.active_index][index_list] = newval

class MeanSquareErrorEvaluator(object):
    def __init__(self):
        self.mse_storage = numpy.empty(100, dtype=numpy.float64)
        self.num_mse = 0
        
    def _evaluate_mse(self, visibility_cache, image, imageparam):
        # UV grid configuration parameters
        uvgrid = imageparam.uvgridconfig
        offsetu = uvgrid.offsetu
        offsetv = uvgrid.offsetv
        nu = uvgrid.nu
        nv = uvgrid.nv
        cellu = uvgrid.cellu
        cellv = uvgrid.cellv
        
        # Obtain visibility from image array
        shifted_image = numpy.fft.fftshift(image)
        shifted_imagefft = numpy.fft2(shifted_image, norm='ortho')
        imagefft = numpy.fft.fftshift(shifted_imagefft)
        rmodel = numpy.flipud(imagefft.real.transpose())
        imodel = numpy.flipud(imagefft.imag.transpose())
        
        # Compute MSE
        mse = 0.0
        num_mse = 0
        rinterp = scipy.interpolate.interp2d(numpy.arange(nv), numpy.arange(nu), rmodel)
        iinterp = scipy.interpolate.interp2d(numpy.arange(nv), numpy.arange(nu), imodel)
        for ws in visibility_cache:
            u = ws.u
            v = ws.v
            rdata = ws.rdata
            idata = ws.idata
            pu = u / cellu + offsetu
            pv = v / cellv + offsetv
            for p, q, x, y in itertools.izip(pu, pv, rdata, idata):
                a = rinterp(q, p)
                b = iinterp(q, p)
                dx = x - a
                dy = y - b
                mse += dx * dx + dy * dy
                num_mse += 1
        mse /= num_mse
        return mse
            
        
    def evaluate_and_accumulate(self, visibility_cache, image, imageparam):
        """
        Evaluate MSE (Mean Square Error) from image which is a solution of MFISTA
        and visibility_cache provided as a set of GridderWorkingSet instance.
        """
        # TODO: evaluate MSE
        mse = self._evaluate_mse(visibility_cache, image, imageparam)
        
        # register it
        if self.num_mse <= len(self.mse_storage):
            self.mse_storage = self.mse_storage.resize(len(self.mse_storage) + 100)
        self.mse_storage[self.num_mse] = mse
        self.num_mse += 1
        
        return mse
    
    def get_mean_mse(self):
        if self.num_mse == 0:
            return 0.0
        else:
            return self.mse_storage[:self.num_mse].mean()

class ApproximateCrossValidationEvaluator(object):
    def __init__(self):
        pass
    
    def evaluate(self, gridvis):
        # TODO: evaluate LOOE (Leave-One-Out Error, aka approximate cross validation)
        return 0.0
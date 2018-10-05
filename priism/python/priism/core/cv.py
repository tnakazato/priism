from __future__ import absolute_import

import numpy
import scipy.interpolate
import contextlib

from . import datacontainer
from . import util
import priism.external.sakura as sakura

class VisibilitySubsetGenerator(object):
    def __init__(self, griddedvis, num_fold=10):
        self.griddedvis = griddedvis
        self.num_fold = num_fold
        
        # Visibility subset is meaningful only when num_fold > 1
        if self.num_fold > 1:
            # amplitude should be nonzero in active pixels
            grid_real = griddedvis.real
            grid_imag = griddedvis.imag
            self.active_index = numpy.where(numpy.logical_or(grid_real != 0, grid_imag != 0))
            self.num_active = len(self.active_index[0])
            print('num_active={0}'.format(self.num_active))
        
            # random index 
            self.index_generator = util.RandomIndexGenerator(self.num_active, self.num_fold)
        else:
            self.active_index = None
            self.num_active = 0
            self.index_generator = None
        
    def get_subset_index(self, subset_id):
        return self.index_generator.get_subset_index(subset_id)
        

class GriddedVisibilitySubsetHandler(object):
    def __init__(self, visset, uvgridconfig):
        # visset is VisibilitySubsetGenerator instance
        assert isinstance(visset, VisibilitySubsetGenerator)
        self.visibility = visset.griddedvis
        self.index_generator = visset.index_generator
        self.active_index = visset.active_index
        self.uvgrid = uvgridconfig
        self.num_fold = visset.num_fold
        
        if self.num_fold <= 1:
            # Visibility subset generator is not properly configured
            raise RuntimeError('VisibilitySubsetGenerator is not properly configured. '
                               + 'Number of visibility subsets is less than 2 ({0})'.format(self.num_fold))
        
        self._clear()
        
    def _clear(self):
        self.visibility_active = None
        self.visibility_cache = None
        #self.active_index = None
        self.subset_id = None
        
        # grid data shape: (nv, nu, npol, nchan)
        grid_real = self.visibility.real
        grid_imag = self.visibility.imag
        
        # amplitude should be nonzero in active pixels
        #self.active_index = numpy.where(numpy.logical_and(grid_real != 0, grid_imag != 0))
        num_active = len(self.active_index[0])
        print('num_active={0}'.format(num_active))
    
        # random index 
        #self.index_generator = util.RandomIndexGenerator(num_active, self.num_fold)
      
    @contextlib.contextmanager
    def generate_subset(self, subset_id):
        self.subset_id = subset_id
        
        # grid data shape: (nv, nu, npol, nchan)
        grid_real = self.visibility.real
        grid_imag = self.visibility.imag
        wgrid_real = self.visibility.wreal
        gdata_shape = grid_real.shape
        
        # random index 
        random_index = self.index_generator.get_subset_index(self.subset_id)
        print('DEBUG_TN: subset ID {0} random_index = {1}'.format(self.subset_id, list(random_index)))
        
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
        num_subvis = len(random_index)
        u = sakura.empty_aligned((num_subvis,), dtype=numpy.float64)
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
        #print 'subset ID {0}: uid{0}={1}; vid{0}={2}'.format(self.subset_id, uid.tolist(), vid.tolist())
        u[:] = (uid - offsetu) * cellu
        v[:] = (vid - offsetv) * cellv
        
        # visibility data to be cached
        # here, we assume npol == 1 (Stokes visibility I_v) and nchan == 1
        assert len(gdata_shape) == 4
        assert gdata_shape[2] == 1 # npol should be 1
        assert gdata_shape[3] == 1 # nchan should be 1 
        real = sakura.empty_aligned((num_subvis,), dtype=numpy.float32)
        imag = sakura.empty_like_aligned(real)
        wreal = sakura.empty_like_aligned(real)
        
        real[:] = grid_real[self.active_index][random_index]
        imag[:] = grid_imag[self.active_index][random_index]
        wreal[:] = wgrid_real[self.active_index][random_index]
        
        # generate subset
        self.visibility_active = self.visibility
        self.__replace_with(self.visibility_active.real, random_index, 0.0)
        self.__replace_with(self.visibility_active.imag, random_index, 0.0)
        self.__replace_with(self.visibility_active.wreal, random_index, 0.0)
        self.visibility_cache = [datacontainer.VisibilityWorkingSet(data_id=0, # nominal data ID
                                                                    rdata=real, 
                                                                    idata=imag,
                                                                    weight=wreal,
                                                                    u=u,
                                                                    v=v)]
        
        try:
            yield self
        finally:
            self.restore_visibility()
        
    def restore_visibility(self):
        if self.visibility_active is not None and self.visibility_cache is not None:
            random_index = self.index_generator.get_subset_index(self.subset_id)
            for cache in self.visibility_cache:
                self.__replace_with(self.visibility.real, random_index, cache.rdata)
                self.__replace_with(self.visibility.imag, random_index, cache.idata)
                self.__replace_with(self.visibility.wreal, random_index, cache.weight)
            self._clear()
            
    def __replace_with(self, src, index_list, newval):
        replace_index = tuple([x[index_list] for x in self.active_index])
        src[replace_index] = newval

class MeanSquareErrorEvaluator(object):
    def __init__(self):
        self.mse_storage = numpy.empty(100, dtype=numpy.float64)
        self.num_mse = 0
        
    def _evaluate_mse(self, visibility_cache, image, uvgrid):
        # UV grid configuration parameters
        offsetu = uvgrid.offsetu
        offsetv = uvgrid.offsetv
        nu = uvgrid.nu
        nv = uvgrid.nv
        cellu = uvgrid.cellu
        cellv = uvgrid.cellv
        
        # Obtain visibility from image array
        image_flipped = numpy.fliplr(image[:,:,0,0])
        shifted_image = numpy.fft.fftshift(image_flipped)
        shifted_imagefft = numpy.fft.fft2(shifted_image)
        imagefft = numpy.fft.ifftshift(shifted_imagefft)
        imagefft_transpose = imagefft.transpose()
        rmodel = imagefft_transpose.real
        imodel = imagefft_transpose.imag
        
        # Compute MSE
        mse = 0.0
        num_terms = 0
        rinterp = scipy.interpolate.interp2d(numpy.arange(nv), numpy.arange(nu), rmodel)
        iinterp = scipy.interpolate.interp2d(numpy.arange(nv), numpy.arange(nu), imodel)
        uid = []
        vid = []
        for ws in visibility_cache:
            u = ws.u
            v = ws.v
            rdata = ws.rdata
            idata = ws.idata
            pu = u / cellu + offsetu
            pv = v / cellv + offsetv
            uid.append(pu)
            vid.append(pv)
            for p, q, x, y in zip(pu, pv, rdata, idata):
                a = rinterp(p, q) # please take care about index order
                b = iinterp(p, q)
                dx = x - a
                dy = y - b
                mse += dx * dx + dy * dy
                num_terms += 1
        mse /= num_terms
        return mse
            
        
    def evaluate_and_accumulate(self, visibility_cache, image, uvgridconfig):
        """
        Evaluate MSE (Mean Square Error) from image which is a solution of MFISTA
        and visibility_cache provided as a set of GridderWorkingSet instance.
        """
        # TODO: evaluate MSE
        mse = self._evaluate_mse(visibility_cache, image, uvgridconfig)
        
        # register it
        if self.num_mse >= len(self.mse_storage):
            self.mse_storage = numpy.resize(self.mse_storage, len(self.mse_storage) + 100)
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

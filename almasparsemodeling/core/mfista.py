from __future__ import absolute_import

import numpy

import almasparsemodeling.external.sakura as sakura
import almasparsemodeling.external.sparseimaging as pysparseimaging

class MfistaSolver(object):
    """
    Solver for sparse modeling using MFISTA algorithm
    """
    def __init__(self, mfistaparam):
        """
        Constructor
        """
        self.mfistaparam = mfistaparam
        
    @property
    def l1(self):
        if hasattr(self, 'mfistaparam'):
            return self.mfistaparam.l1
        else:
            return None
    
    @property
    def ltsv(self):
        if hasattr(self, 'mfistaparam'):
            return self.mfistaparam.ltsv
        else:
            return None
            
    def solve(self, grid_data):
        """
        Given complex visibility data, find the best image 
        under the condition of 
        
            min{sum[(Fxi - yi)^2] + L1 |xi| + Ltsv TSV(x)} 
        
        Solve the problem using MFISTA algorithm.
        """
        image_shape = grid_data.shape
        image_data = sakura.empty_aligned(image_shape, dtype=numpy.float64)
        sakura.solvemfista(self.l1, self.ltsqv, grid_data, image_data)
        
class MfistaSolverExternal(MfistaSolver):
    """
    Solver for sparse modeling using MFISTA algorithm
    
    This depends on sparseimaging package written by Shiro Ikeda. 
    It calls C-function via wrapper class defined in external submodule.
    (external.sparseimaging.executor.SparseImagingExecutor)
    """
    def __init__(self, mfistaparam, libpath=None):
        """
        Constructor
        """
        super(MfistaSolverExternal, self).__init__(mfistaparam)
        self.libpath = libpath
        
    def solve(self, grid_data):
        """
        Given complex visibility data, find the best image 
        under the condition of 
        
            min{sum[(Fxi - yi)^2] + L1 |xi| + Ltsv TSV(x)} 
        
        Solve the problem using MFISTA algorithm.
        """
        # TODO: nonnegative must be specified by the user
        executor = pysparseimaging.SparseImagingExecutor(lambda_L1=self.l1,
                                                         lambda_TSV=self.ltsv,
                                                         nonnegative=True,
                                                         libpath=self.libpath)
        # TODO: define converter from gridded data to inputs
        inputs = pysparseimaging.SparseImagingInputs.from_gridder_result(grid_data)

        result = executor.run(inputs)
        
        image = result.image
        # add degenerate axis (polarization and spectral)
        image = image.reshape((image.shape[0], image.shape[0], 1, 1))
        
        # flip longitude 
        image = numpy.fliplr(image)

        return image
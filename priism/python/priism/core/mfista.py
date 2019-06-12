from __future__ import absolute_import

import numpy

from . import sparseimagingfft
from . import sparseimagingnufft


class MfistaSolverBase(object):
    """
    Solver for sparse modeling using MFISTA algorithm
    """
    def __init__(self, mfistaparam, imageparam):
        """
        Constructor
        """
        self.mfistaparam = mfistaparam
        self.imageparam = imageparam
        self.initialimage = None

    def __from_param(self, name):
        if hasattr(self, 'mfistaparam'):
            return getattr(self.mfistaparam, name)
        else:
            None

    @property
    def l1(self):
        return self.__from_param('l1')

    @property
    def ltsv(self):
        return self.__from_param('ltsv')

    @property
    def maxiter(self):
        return self.__from_param('maxiter')

    @property
    def eps(self):
        return self.__from_param('eps')

    @property
    def clean_box(self):
        return self.__from_param('clean_box')

    @property
    def box_flag(self):
        return 0 if self.clean_box is None else 1

    @property
    def imsize(self):
        return self.imageparam.imsize

    @property
    def nx(self):
        return self.imsize[0]

    @property
    def ny(self):
        return self.imsize[1]

    def solve(self, grid_data):
        """
        Given complex visibility data, find the best image
        under the condition of

            min{sum[(Fxi - yi)^2] + L1 |xi| + Ltsv TSV(x)}

        Solve the problem using MFISTA algorithm.
        """
        raise NotImplementedError('solve method shold be defined in each subclass.')


class MfistaSolverTemplate(MfistaSolverBase):
    """
    Template class for sparse modeling solver.
    """
    Executor = None

    def __init__(self, mfistaparam, imageparam):
        """
        Constructor
        """
        super(MfistaSolverTemplate, self).__init__(mfistaparam, imageparam)

    def solve(self, visibility, imageparam, storeinitialimage=True, overwriteinitialimage=False):
        """
        Given complex visibility data, find the best image
        under the condition of

            min{sum[(Fxi - yi)^2] + L1 |xi| + Ltsv TSV(x)}
        
        Solve the problem using MFISTA algorithm.

        visibility -- visibility data
        imageparam -- image parameter
        """
        assert self.Executor is not None

        # TODO: nonnegative must be specified by the user
        executor = self.Executor(lambda_L1=self.l1,
                                 lambda_TSV=self.ltsv,
                                 nonnegative=True)
        inputs = executor.Inputs.from_visibility_working_set(visibility,
                                                             imageparam)

        result = executor.run(inputs, initialimage=self.initialimage,
                              maxiter=self.maxiter, eps=self.eps, cl_box=self.clean_box)

        # keep output image as an initial image to next run if necessary
        if storeinitialimage:
            if self.initialimage is None or overwriteinitialimage:
                self.initialimage = result.xout.copy()

        # normalize resulting image
        self.normalize_result(inputs, result)

        image = result.image
        # add degenerate axis (polarization and spectral)
        image = image.reshape((image.shape[0], image.shape[1], 1, 1))

        return image

    def normalize_result(self, vis_data, image_data):
        """
        Normalize resulting image. Do nothing by default.

        vis_data -- input visiblity as the form of SparseImagingInputs
        image_data -- output image as the form of SparseImagingResults
        """
        pass


class SakuraSolver(MfistaSolverBase):
    def __init__(self, mfistaparam, imageparam):
        super(SakuraSolver, self).__init__(mfistaparam, imageparam)

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


class MfistaSolverFFT(MfistaSolverTemplate):
    """
    Solver for sparse modeling using MFISTA algorithm with FFT

    This depends on sparseimaging package written by Shiro Ikeda. 
    It calls C-function via wrapper class defined in external submodule.
    (priism.core.sparseimagingfft.SparseImagingExecutor)
    """
    Executor = sparseimagingfft.SparseImagingExecutor

    def __init__(self, mfistaparam, imageparam):
        """
        Constructor
        """
        super(MfistaSolverFFT, self).__init__(mfistaparam, imageparam)

    def normalize_result(self, vis_data, image_data):
        """
        Normalize resulting image. Multiply sqrt(Nx*Ny) to conpensate for 
        the difference of normalization strategy of FFT.

        vis_data -- input visiblity as the form of SparseImagingInputs
        image_data -- output image as the form of SparseImagingResults
        """
        nx = image_data.nx
        ny = image_data.ny
        factor = 1.0 / numpy.sqrt(nx * ny)
        print('Normalization factor is {}'.format(factor))
        image_data.xout *= factor


class MfistaSolverNUFFT(MfistaSolverTemplate):
    """
    Solver for sparse modeling using MFISTA algorithm with NUFFT

    This depends on sparseimaging package written by Shiro Ikeda. 
    It calls C-function via wrapper class defined in external submodule.
    (priism.core.sparseimagingnufft.SparseImagingExecutor)
     """
    Executor = sparseimagingnufft.SparseImagingExecutor

    def __init__(self, mfistaparam, imageparam):
        """
         Constructor
        """
        super(MfistaSolverNUFFT, self).__init__(mfistaparam, imageparam)


def SolverFactory(mode='mfista_fft'):
    if mode == 'mfista_fft':
        return MfistaSolverFFT
    elif mode == 'mfista_nufft':
        return MfistaSolverNUFFT
    elif mode == 'sakura':
        return SakuraSolver
    else:
        RuntimeError('Unsupported mode: {}'.format(mode))
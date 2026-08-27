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

from argparse import ArgumentError
import collections
import itertools
import logging
import math
import os
import pickle
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna

from . import datacontainer
from . import paramcontainer
from . import mfista
from . import cv
from . import uvcriteria


logger = logging.getLogger(__name__)


def format_lambda(v):
    s = None
    if v < 0:
        s = 'Minus'
    elif v == 0:
        s = 'Zero'
    else:
        s = '{:.2f}'.format(float(math.log10(v)))
    return s


def format_tick(x, value_list):
    v = value_list[int(x)]
    return format_lambda(v)


class SparseModelingImager(object):
    """
    Core implementation of sparse modeling specialized for ALMA.
    It performs visibility gridding on uv-plane.
    """

    CrossValidationResult = collections.namedtuple(
        'CrossValidationResult',
        ['mse', 'image', 'L1', 'Ltsv']
    )

    @property
    def griddedvis(self):
        """
        TODO
        """
        return getattr(self, '_griddedvis', None)

    @griddedvis.setter
    def griddedvis(self, value):
        if value is None:
            self._griddedvis = None
        elif isinstance(value, datacontainer.GriddedVisibilityStorage):
            self._griddedvis = value
        else:
            raise TypeError('Given value is not an instance of GriddedVisibilityStorage')

    @property
    def working_set(self):
        """
        """
        return getattr(self, '_working_set', None)

    @working_set.setter
    def working_set(self, value):
        if value is None:
            self._working_set = None
        elif isinstance(value, datacontainer.VisibilityWorkingSet):
            self._working_set = value
        else:
            raise TypeError('Given value is an instance of VisibilityWorkingSet')

    @property
    def imagearray(self):
        """
        TODO
        """
        return getattr(self, '_imagearray', None)

    @imagearray.setter
    def imagearray(self, value):
        if value is None:
            self._imagearray = None
        elif isinstance(value, datacontainer.ResultingImageStorage):
            self._imagearray = value
        else:
            raise TypeError('Given value is not an instance of ResultingImageStorage')


    @property
    def imagesuffix(self):
        """
        Suffix for the exported image. Default is 'pickle' since the export format is
        Python's cPickle object. Suffix can be customized depending on how to override
        exportimage method.
        """
        return 'pickle'

    def __init__(self, solver='mfista_fft'):
        """
        Constructor

        Parameters:
            solver  name of the solver
                    choices are as follows.
                      'mfista_fft'    MFISTA algorithm with FFT by S. Ikeda.
                      'mfista_nufft'  MFISTA algorithm with NUFFT by S. Ikeda
                                       (to be implemented in future)
       """
        self.solver_name = solver
        self.__initialize()

    def __initialize(self):
        # configuration
        self.imparam = None
        self.visparams = []

        # working array
        self.griddedvis = None
        self.imagearray = None

        # TODO: optimize number of threads
        self.num_threads = 2

        # create MFISTA instance with dummy parameter
        mfistaparam = paramcontainer.MfistaParamContainer(l1=0.0, ltsv=0.0)
        solver_cls = mfista.SolverFactory(self.solver_name)
        self.solver = solver_cls(mfistaparam)

    def mfista(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None,
               storeinitialimage=True, overwriteinitialimage=False, nonnegative=True,
               nthreads=1):
        logger.warning('***WARNING*** mfista will be deprecate in the future. Please use solve instead.')
        self.solve(l1, ltsv, maxiter, eps, clean_box,
                   storeinitialimage, overwriteinitialimage, nonnegative,
                   nthreads=nthreads)

    def solve(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None,
              storeinitialimage=True, overwriteinitialimage=False, nonnegative=True,
              scalehyperparam=True, nthreads=1):
        """
        Run MFISTA algorithm on visibility data loaded on memory.
        gridvis or readvis must be executed beforehand.

        Parameters:
            l1 -- L1 regularization term
            ltsv -- TSV regularization term
            maxiter -- maximum number of iteration for MFISTA
            eps -- threshold factor for MFISTA
            clean_box -- clean box as a float array
            storeinitialimage -- keep the result as an initial image for next run
            overwriteinitialimage -- overwrite existing initial image
            nonnegative -- allow negative value (False) or not (True)
            scalehyperparam -- apply hyper-parameter scaling (L1 and Ltsv) to reproduce
                               the behavior compatible with previous version (earlier than
                               0.9.x). Default is True (backward-compatible).
            nthreads -- number of threads finufft may use per NUFFT call, when
                        solver='pymfista_nufft'. Default is 1 to avoid
                        oversubscribing when multiple solves run concurrently
                        (e.g. cross-validation grid search); raise it if a
                        single solve has the machine to itself. Ignored by
                        the C++-based solvers.
        """
        if scalehyperparam:
            # scaling factor for hyper-parameter
            hp_scale = 2.0 / np.sqrt(self.imparam.imsize[0] * self.imparam.imsize[1])
            internal_L1 = l1 * hp_scale
            internal_Ltsv = ltsv * hp_scale * hp_scale
        else:
            internal_L1 = l1
            internal_Ltsv = ltsv

        self.mfistaparam = paramcontainer.MfistaParamContainer(l1=internal_L1, ltsv=internal_Ltsv,
                                                               maxiter=maxiter, eps=eps,
                                                               clean_box=clean_box,
                                                               nonnegative=nonnegative,
                                                               nthreads=nthreads)
        arr = self._solve(self.mfistaparam, self.working_set,
                          storeinitialimage=storeinitialimage, overwriteinitialimage=overwriteinitialimage)
        self.imagearray = datacontainer.ResultingImageStorage(arr)

    def _solve(self, mfistaparam, working_set, storeinitialimage=True, overwriteinitialimage=False):
        assert working_set is not None
        if working_set is None or len(working_set.rdata) == 0:
            raise RuntimeError(
                "No visibility data is given. Please run readvis or gridvis first. "
                "If you already did, probably all the visibility data are flagged (invalid). "
                "Please check contents of the MS and tweak channel selection appropriately."
            )
        self.solver.mfistaparam = mfistaparam
        return self.solver.solve(working_set, self.imparam, storeinitialimage, overwriteinitialimage)

    def importvis(self, data=None, weight=None, filename=None, flipped=False):
        """
        Import visibility data. Users can provide visibility data either as numpy array
        (data and weight) or as filename that stores visibility data in a specified format.
        Either data or filename should be specified. If both are specified, filename takes
        priority.

        Parameters:
            data     -- Visibility data as numpy complex array. Its shape must effectively
                        be two-dimensional (nv, nu). For unflipped array, total power component
                        (0,0) must be located at (nv//2, nu//2). For flipped array, order of
                        the array elements follows convention of FFTW3 library.
                        Additional axes (spectral and stokes) may be added but their length
                        must be 1.
            weight   -- Visibility weight (inverse square of sigma) as numpy array. Array
                        type can be either float or complex. If complex array is given,
                        its real part is interpreted as a weight for real part of the
                        visibility while the imaginary part is a weight for imaginary part.
                        Array shape must conform with data. None is also acceptable.
                        In this case, equal weight (1.0) will be applied to all visibilities.
            filename -- Name of the file that stores visibility data and weights. Format
                        should be as follows:
            flipped  -- Whether or not given data and weight are flipped for FFT.
        """
        if data is None and filename is None:
            raise RuntimeError('data or filename must be specified')

        if filename is not None:
            # filename is specified. read it.
            griddedvis = datacontainer.GriddedVisibilityStorage.importdata(filename)
            realdata = griddedvis.real
            imagdata = griddedvis.imag
            realweight = griddedvis.wreal
            imagweight = griddedvis.wimag
            default_nu = realdata.shape[1]
            default_nv = realdata.shape[0]
        else:
            datashape = data.shape
            default_nu = datashape[1]
            default_nv = datashape[0]
            if weight is None:
                # use default weight (all 1.0)
                weight = np.ones(datashape, dtype=np.float32)

            weightshape = weight.shape
            if datashape != weightshape:
                raise RuntimeError('Array shape of weight must conform with that of data.')

            if len(datashape) == 2:
                newdata = np.expand_dims(np.expand_dims(data, axis=-1), axis=-1)
                newweight = np.reshape(weight, newdata.shape)
            elif len(datashape) == 3:
                if datashape[2] > 1:
                    raise RuntimeError('Invalid array shape {}'.format(list(datashape)))
                newdata = np.expand_dims(data, axis=-1)
                newweight = np.reshape(weight, newdata.shape)
            elif len(datashape) == 4:
                if datashape[2] > 1 or datashape[3] > 1:
                    raise RuntimeError('Invalid array shape {}'.format(list(datashape)))
                newdata = data
                newweight = weight
            else:
                raise RuntimeError('Invalid array shape {}'.format(list(datashape)))

            if newdata.dtype not in (np.complex, complex):
                raise TypeError('data must be float complex array')

            realdata = newdata.real
            imagdata = newdata.imag

            if newweight.dtype not in (np.float32, np.float, float, np.complex, complex):
                raise TypeError('weight must be float or float complex array')

            if newweight.dtype in (np.float32, np.float, float):
                realweight = newweight
                imagweight = None
            else:
                realweight = newweight.real
                imagweight = newweight.imag

        # flip back operation if necessary
        if flipped is True:
            realdata = np.fft.fftshift(realdata)
            imagdata = np.fft.fftshift(imagdata)
            realweight = np.fft.fftshift(realweight)
            if imagweight is not None:
                imagweight = np.fft.fftshift(imagweight)

        self.griddedvis = datacontainer.GriddedVisibilityStorage(grid_real=realdata,
                                                                 grid_imag=imagdata,
                                                                 wgrid_real=realweight,
                                                                 wgrid_imag=imagweight)
        self.working_set = datacontainer.grid2ws(realdata, imagdata, realweight, imagweight)

        self.imparam = paramcontainer.SimpleImageParamContainer(imsize=[default_nu, default_nv])

    def exportimage(self, imagename, overwrite=False):
        """
        Export MFISTA result as a cPickle object.
        mfista must be executed beforehand.

        Parameters:
            imagename  name of output image name
        """
        if self.imagearray is None:
            raise RuntimeError('You don\'t have an image array!')

        if os.path.exists(imagename) and overwrite is False:
            raise RuntimeError('Cannot overwrite existing file "{}"'.format(imagename))

        with open(imagename, 'wb') as f:
            pickle.dump(self.imagearray, f)

    def getimage(self, imagename):
        """
        Get image data from exported file

        Parameters:
            imagename  name of image to be read
        """
        if not os.path.exists(imagename):
            raise RuntimeError('image "{}" does not exist'.format(imagename))

        with open(imagename, 'rb') as f:
            data = pickle.load(f)

        # exportimage() pickles self.imagearray, which is already a
        # ResultingImageStorage instance -- return the unpickled object
        # directly rather than wrapping it in a second ResultingImageStorage
        # (that double-wrap left .data holding a ResultingImageStorage
        # instead of a plain ndarray).
        return data

    def cvforgridvis(self, l1_list, ltsv_list, num_fold=10, imageprefix='image', imagepolicy='full',
                     summarize=True, figfile=None, datafile=None, maxiter=50000, eps=1.0e-5, clean_box=None,
                     resultasinitialimage=True, nonnegative=True):
        logger.warning('***WARNING*** cvforgridvis will be deprecate in the future. Please use optimizeparameters instead.')
        return self.crossvalidation(l1_list, ltsv_list, num_fold, imageprefix, imagepolicy,
                                    summarize, figfile, datafile, maxiter, eps, clean_box,
                                    resultasinitialimage, nonnegative=True, )

    def optimizeparameters(self, l1_list, ltsv_list, num_fold=10, imageprefix='image', imagepolicy='full',
                        summarize=True, figfile=None, datafile=None, maxiter=50000, eps=1.0e-5, clean_box=None,
                        resultasinitialimage=True, nonnegative=True, scalehyperparam=True,
                        criterion='cv', optimizer='classical',
                        bayesopt_maxiter=15, ellipse_th=0.99, cos_th=0.99,
                        bayesopt_n_startup_trials=None):
        """
        Search the best parameter for L1 and Ltsv from the given list of these.

        The search is controlled by two independent choices:
            criterion -- how a given (L1, Ltsv) is scored. 'cv' (default) uses
                         cross-validation MSE on held-out visibility subsets.
                         'ellipsoid' uses the u-v-distance-grouped criterion from
                         Ikeda et al. 2025 (PASJ 77(2):260-276, section 3.4):
                         the minimum covering u-v ellipsoid power ratio (C1) and
                         grouped-residual cosine similarity (C2) are treated as
                         soft constraints (C1 >= ellipse_th, C2 >= cos_th), and
                         among (L1, Ltsv) satisfying both, the weighted mean-
                         squared visibility residual is minimized. This is
                         computed from a single full-data MFISTA solve per
                         point, so no cross-validation subsetting is performed
                         and num_fold is ignored.
            optimizer -- how (L1, Ltsv) space is searched. 'classical' (default)
                         is an exhaustive grid search over l1_list x ltsv_list.
                         'bayesian' uses Optuna to adaptively pick points from
                         the same grid (see bayesopt_maxiter).

        Inputs:
            l1_list -- list of L1 values to examine
            ltsv_list -- List of Ltsv values to examine
            num_fold -- number of visibility subsets for cross validation
                        (only used when criterion='cv')
            imageprefix -- prefix for output image
                           imageprefix is used for the best image (<imageprefix>.fits)
            imagepolicy -- policy of output image ('full' or 'best')
                           full: keep all FITS image produced by cross validation
                           best: only keep FITS image corersponding to the best solution
            summarize -- generate summary plot if True
            figfile -- name of summary figure of cross validation.
                       None will not produce a file.
            datafile -- name of output data file containing whole MSE values.
                        None will not produce a file.
            maxiter -- maximum number of iteration for MFISTA algorithm
            eps -- threshold factor for MFISTA algorithm
            clean_box -- clean box as a float array (default None)
            resultasinitialimage -- keep resulting image as an initial condition for next run
            nonnegative -- allow negative value (False) or not (True)
            scalehyperparam -- apply hyper-parameter scaling (L1 and Ltsv) to reproduce
                               the behavior compatible with previous version (earlier than
                               0.9.x). Default is True (backward-compatible).
            criterion -- evaluation criterion. 'cv' or 'ellipsoid'. See above.
            optimizer -- search strategy. 'classical' or 'bayesian'. See above.
            bayesopt_maxiter -- (specific to optimizer='bayesian')
            bayesopt_n_startup_trials -- (specific to optimizer='bayesian') number
                          of purely-random trials before Optuna's TPE sampler
                          starts using its surrogate model (Optuna's own default
                          is 10). If bayesopt_maxiter is close to or smaller than
                          this, the search is effectively random with no real
                          Bayesian guidance ever applied -- for a small
                          bayesopt_maxiter budget, consider lowering this too.
                          None (default) keeps Optuna's own default.
            ellipse_th -- (specific to criterion='ellipsoid') soft-constraint threshold
                          for C1 (minimum covering u-v ellipsoid power ratio). Default 0.99.
            cos_th -- (specific to criterion='ellipsoid') soft-constraint threshold for C2
                      (grouped-residual cosine similarity). Default 0.99.

        Output:
            dictionary containing best L1 (key: L1), best Ltsv (key;Ltsv), and
            corresponding image name (key: image, should be <imageprefix>.fits)
        """
        start_time = time.time()

        # sanity check
        if imagepolicy not in ('best', 'full'):
            raise ArgumentError('imagepolicy must be best or full. {0} was provided.'.format(imagepolicy))
        if criterion not in ('cv', 'ellipsoid'):
            raise ArgumentError("criterion should be 'cv' or 'ellipsoid'")
        if optimizer not in ('classical', 'bayesian'):
            raise ArgumentError("optimizer should be 'classical' or 'bayesian'")

        try:
            np_l1_list = np.asarray(l1_list)
            np_ltsv_list = np.asarray(ltsv_list)
        except Exception as e:
            logger.error('Exception occurred')
            logger.error(str(e))
            raise ArgumentError('l1_list or ltsv_list (or both) seems invalid.')

        if str(np_l1_list.dtype) == 'object':
            raise ArgumentError('l1_list contains invalid value')
        if str(np_ltsv_list.dtype) == 'object':
            raise ArgumentError('ltsv_list contains invalid value')

        L1_sort_index = np.argsort(np_l1_list)
        Ltsv_sort_index = np.argsort(np_ltsv_list)

        sorted_l1_list = np_l1_list[L1_sort_index]
        sorted_ltsv_list = np_ltsv_list[Ltsv_sort_index]

        # initialize CV (not needed for criterion='ellipsoid', which evaluates on the full data)
        if criterion == 'cv':
            self.initializecv(num_fold=num_fold)

        # scaling factor for hyper-parameter
        if scalehyperparam:
            hp_scale = 2.0 / np.sqrt(self.imparam.imsize[0] * self.imparam.imsize[1])
        else:
            hp_scale = 1.0

        if criterion == 'cv' and optimizer == 'classical':
            result = self._cv_classical(
                l1_list=sorted_l1_list, ltsv_list=sorted_ltsv_list, hp_scale=hp_scale,
                imageprefix=imageprefix, maxiter=maxiter, eps=eps, clean_box=clean_box,
                nonnegative=nonnegative, resultasinitialimage=resultasinitialimage,
            )
        elif criterion == 'cv' and optimizer == 'bayesian':
            result = self._cv_bayesian(
                l1_list=sorted_l1_list, ltsv_list=sorted_ltsv_list, hp_scale=hp_scale,
                imageprefix=imageprefix, maxiter=maxiter, eps=eps, clean_box=clean_box,
                nonnegative=nonnegative, resultasinitialimage=resultasinitialimage,
                bayesopt_maxiter=bayesopt_maxiter,
                bayesopt_n_startup_trials=bayesopt_n_startup_trials
            )
        elif criterion == 'ellipsoid' and optimizer == 'classical':
            result = self._ellipsoid_classical(
                l1_list=sorted_l1_list, ltsv_list=sorted_ltsv_list, hp_scale=hp_scale,
                imageprefix=imageprefix, maxiter=maxiter, eps=eps, clean_box=clean_box,
                nonnegative=nonnegative, resultasinitialimage=resultasinitialimage,
                ellipse_th=ellipse_th, cos_th=cos_th
            )
        elif criterion == 'ellipsoid' and optimizer == 'bayesian':
            result = self._ellipsoid_bayesian(
                l1_list=sorted_l1_list, ltsv_list=sorted_ltsv_list, hp_scale=hp_scale,
                imageprefix=imageprefix, maxiter=maxiter, eps=eps, clean_box=clean_box,
                nonnegative=nonnegative, resultasinitialimage=resultasinitialimage,
                bayesopt_maxiter=bayesopt_maxiter, ellipse_th=ellipse_th, cos_th=cos_th,
                bayesopt_n_startup_trials=bayesopt_n_startup_trials
            )
        else:
            assert False, 'unreachable (criterion/optimizer already validated above)'

        # finalize CV
        if criterion == 'cv':
            self.finalizecv()

        best_solution = np.argmin(result.mse)
        best_mse = result.mse[best_solution]
        best_image = result.image[best_solution]
        best_L1 = result.L1[best_solution]
        best_Ltsv = result.Ltsv[best_solution]

        if datafile is not None:
            with open(datafile, 'w') as f:
                print('# L1, Ltsv, MSE', file=f)
                for mse, _, L1, Ltsv in zip(*result):
                    print(f'{L1}, {Ltsv}, {mse}', file=f)

        if summarize:
            self._plot_cv_result(
                sorted_l1_list, sorted_ltsv_list, result, best_solution, figfile=figfile,
                optimizer=optimizer
            )


        # completed
        end_time = time.time()

        if best_mse >= 0.0:
            logger.info('Process completed. Optimal result is as follows')
            L1str = '{}'.format(f'10^{int(math.log10(best_L1))}' if best_L1 > 0 else format_lambda(best_L1))
            Ltsvstr = '{}'.format(f'10^{int(math.log10(best_Ltsv))}' if best_Ltsv > 0 else format_lambda(best_Ltsv))
            logger.info(f'    L1, Ltsv = {L1str}, {Ltsvstr}')
            logger.info(f'    MSE = {best_mse}')
            logger.info(f'    imagename = {best_image}')
        else:
            logger.info('Process completed. Cross-validation was not performed.')
            logger.warning('WARNING: Optimal solution will not be correct one since no CV was executed.')

        logger.info(f'Elapsed {end_time - start_time} sec')

        # copy the best image to final image
        shutil.copy2(best_image, imageprefix + '.' + self.imagesuffix)
        if imagepolicy == 'full':
            # keep all intermediate images
            pass
        elif imagepolicy == 'best':
            # remove all intermediate images. A single (L1, Ltsv) grid point
            # can appear more than once in result.image -- optimizer='bayesian'
            # can revisit the same point across trials, and each such trial
            # writes to the same file (imagename is derived purely from
            # L1/Ltsv) -- so remove each unique filename once instead of
            # raising FileNotFoundError on the repeat.
            for imagename in set(result.image):
                os.remove(imagename)
        else:
            assert False

        # finally, return best L1 and Ltsv
        return {'L1': best_L1, 'Ltsv': best_Ltsv}

    def crossvalidation(self, l1_list, ltsv_list, num_fold=10, imageprefix='image', imagepolicy='full',
                        summarize=True, figfile=None, datafile=None, maxiter=50000, eps=1.0e-5, clean_box=None,
                        resultasinitialimage=True, nonnegative=True, scalehyperparam=True, optimizer='classical',
                        bayesopt_maxiter=15):
        """
        Deprecated. Use optimizeparameters(..., criterion='cv', optimizer=...) instead.
        Kept as a thin wrapper with its original (pre-'ellipsoid'-criterion)
        signature, always selecting criterion='cv'.
        """
        logger.warning(
            '***WARNING*** crossvalidation will be deprecated in the future. '
            'Please use optimizeparameters instead.'
        )
        return self.optimizeparameters(
            l1_list, ltsv_list, num_fold, imageprefix, imagepolicy,
            summarize, figfile, datafile, maxiter, eps, clean_box,
            resultasinitialimage, nonnegative, scalehyperparam,
            criterion='cv', optimizer=optimizer, bayesopt_maxiter=bayesopt_maxiter
        )

    def initializecv(self, num_fold=10):
        assert self.working_set is not None

        if (not hasattr(self, 'visset')) or self.visset is None:
            self.visset = cv.VisibilitySubsetGenerator(self.working_set, num_fold)

    def finalizecv(self):
        self.visset = None

    def _cv_exec(self, l1, ltsv, hp_scale, imageprefix='image',
                 maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                 resultasinitialimage=True, overwriteinitialimage=True):
        """
        Evaluate a single (l1, ltsv) point via cross-validation MSE.
        Signature matches the "exec_fn(l1, ltsv, overwrite_initial=True) ->
        (cost, imagename)" contract expected by _search_classical /
        _search_bayesian.
        """
        # get full visibility image first
        l1_str = format_lambda(l1)
        ltsv_str = format_lambda(ltsv)
        imagename = f'{imageprefix}_L1_{l1_str}_Ltsv_{ltsv_str}.{self.imagesuffix}'

        internal_l1 = l1 * hp_scale
        internal_ltsv = ltsv * hp_scale * hp_scale

        self.solve(internal_l1, internal_ltsv,
                   maxiter=maxiter, eps=eps, clean_box=clean_box,
                   nonnegative=nonnegative,
                   storeinitialimage=resultasinitialimage,
                   overwriteinitialimage=overwriteinitialimage,
                   scalehyperparam=False)
        self.exportimage(imagename, overwrite=True)

        # then evaluate MSE
        mse = self.computemse(internal_l1, internal_ltsv, maxiter, eps, clean_box, nonnegative=nonnegative)

        logger.info(f'L1 10^{l1_str} Ltsv 10^{ltsv_str}: MSE {mse} FITS {imagename}')

        return mse, imagename

    def _cv_exec_with_plateau_scaling(self, l1, ltsv, hp_scale, imageprefix='image',
                                       maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                                       resultasinitialimage=True):
        """
        Same as _cv_exec, but scales the MSE up when the resulting image is
        empty (all-zero), to avoid Bayesian Optimization wasting trials on
        the flat "empty image" MSE plateau caused by too strong an L1
        constraint. Used by the 'bayesian' search only (the 'classical'
        grid search has no such issue, since it visits every grid point
        regardless).
        """
        mse, imagename = self._cv_exec(
            l1, ltsv, hp_scale, imageprefix, maxiter,
            eps, clean_box, nonnegative, resultasinitialimage
        )

        data_storage = self.getimage(imagename)
        data = data_storage.data
        if np.all(data == 0):
            factor = 1 + max(2, math.log10(l1)) / 10
            mse *= factor

        return mse, imagename

    def _ellipsoid_exec(self, l1, ltsv, hp_scale, imageprefix='image',
                        maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                        resultasinitialimage=True, overwriteinitialimage=True,
                        evaluator=None, ellipse_th=0.99, cos_th=0.99):
        """
        Evaluate a single (l1, ltsv) point via the u-v-distance-grouped
        criterion (uvcriteria.UvEllipsoidEvaluator), on the full (non-CV-
        split) working set. Same "exec_fn" contract as _cv_exec.

        C1 (u-v ellipsoid power ratio) and C2 (grouped-residual cosine
        similarity) are treated as soft constraints (C1 >= ellipse_th,
        C2 >= cos_th): among (L1, Ltsv) that satisfy both, the one with the
        smallest weighted mean-squared visibility residual is preferred.
        """
        assert evaluator is not None

        l1_str = format_lambda(l1)
        ltsv_str = format_lambda(ltsv)
        imagename = f'{imageprefix}_L1_{l1_str}_Ltsv_{ltsv_str}.{self.imagesuffix}'

        self.solve(l1 * hp_scale, ltsv * hp_scale * hp_scale,
                   maxiter=maxiter, eps=eps, clean_box=clean_box,
                   nonnegative=nonnegative,
                   storeinitialimage=resultasinitialimage,
                   overwriteinitialimage=overwriteinitialimage,
                   scalehyperparam=False)
        self.exportimage(imagename, overwrite=True)

        # Use the freshly-computed self.imagearray directly rather than
        # round-tripping through getimage(imagename)/disk -- exportimage()
        # is still called above so the FITS/pickle file exists on disk for
        # the imagepolicy/best-image handling in optimizeparameters().
        image_2d = np.squeeze(self.imagearray.data)
        cost, c1, c2 = evaluator.evaluate(
            self.working_set, image_2d, ellipse_th=ellipse_th, cos_th=cos_th
        )

        logger.info(f'L1 10^{l1_str} Ltsv 10^{ltsv_str}: cost {cost} (C1={c1}, C2={c2}) FITS {imagename}')

        return cost, imagename

    def _search_classical(self, l1_list, ltsv_list, exec_fn):
        """
        Exhaustive grid search over (l1_list, ltsv_list), evaluating each
        point with exec_fn(l1, ltsv, overwrite_initial) -> (cost, imagename).
        """
        result_L1 = []
        result_Ltsv = []
        result_mse = []
        result_image = []

        # loop Ltsv in ascending order
        for j, Ltsv in enumerate(ltsv_list):
            # trick to update initial image when Ltsv is changed
            overwrite_initial = True

            # loop L1 in descending order
            for i in range(len(l1_list) - 1, -1, -1):
                L1 = l1_list[i]
                result_L1.append(L1)
                result_Ltsv.append(Ltsv)

                cost, imagename = exec_fn(L1, Ltsv, overwrite_initial)

                result_image.append(imagename)
                result_mse.append(cost)

                overwrite_initial = False

        return self.CrossValidationResult(
            mse=result_mse, image=result_image,
            L1=result_L1, Ltsv=result_Ltsv
        )

    def _search_bayesian(self, l1_list, ltsv_list, exec_fn, bayesopt_maxiter=15,
                         bayesopt_n_startup_trials=None):
        """
        Bayesian Optimization (Optuna) search over indices into
        (l1_list, ltsv_list), evaluating each trial with
        exec_fn(l1, ltsv) -> (cost, imagename).

        bayesopt_n_startup_trials -- number of purely-random trials Optuna's
                     default TPESampler runs before it starts using its
                     surrogate model to guide sampling (Optuna's own
                     TPESampler default is 10). If bayesopt_maxiter is close
                     to or smaller than this, the search degenerates to
                     random sampling with no actual Bayesian guidance ever
                     applied. None (the default here) keeps Optuna's own
                     default unchanged.
        """
        result_L1 = []
        result_Ltsv = []
        result_mse = []
        result_image = []

        def objective(trial):
            L1_index = trial.suggest_int("L1 index", 0, len(l1_list) - 1)
            L1 = l1_list[L1_index]
            Ltsv_index = trial.suggest_int("Ltsv index", 0, len(ltsv_list) - 1)
            Ltsv = ltsv_list[Ltsv_index]

            cost, imagename = exec_fn(L1, Ltsv)

            result_L1.append(L1)
            result_Ltsv.append(Ltsv)
            result_mse.append(cost)
            result_image.append(imagename)

            return cost

        if bayesopt_n_startup_trials is not None:
            sampler = optuna.samplers.TPESampler(n_startup_trials=bayesopt_n_startup_trials)
        else:
            sampler = None
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=bayesopt_maxiter)
        self.cv_bayes_result = study.best_params

        return self.CrossValidationResult(
            mse=result_mse, image=result_image,
            L1=result_L1, Ltsv=result_Ltsv
        )

    def _cv_classical(self, l1_list, ltsv_list, hp_scale=1.0, imageprefix='image',
                      maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                      resultasinitialimage=True):
        def exec_fn(l1, ltsv, overwrite_initial):
            return self._cv_exec(
                l1, ltsv, hp_scale, imageprefix, maxiter,
                eps, clean_box, nonnegative, resultasinitialimage,
                overwrite_initial
            )
        return self._search_classical(l1_list, ltsv_list, exec_fn)

    def _cv_bayesian(self, l1_list, ltsv_list, hp_scale=1.0, num_fold=10, imageprefix='image',
                      maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                      resultasinitialimage=True, bayesopt_maxiter=15,
                      bayesopt_n_startup_trials=None):
        def exec_fn(l1, ltsv):
            return self._cv_exec_with_plateau_scaling(
                l1, ltsv, hp_scale, imageprefix, maxiter,
                eps, clean_box, nonnegative, resultasinitialimage
            )
        return self._search_bayesian(l1_list, ltsv_list, exec_fn, bayesopt_maxiter,
                                     bayesopt_n_startup_trials)

    def _ellipsoid_classical(self, l1_list, ltsv_list, hp_scale=1.0, imageprefix='image',
                             maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                             resultasinitialimage=True, ellipse_th=0.99, cos_th=0.99):
        evaluator = uvcriteria.UvEllipsoidEvaluator(
            self.working_set, self.imparam.imsize[0], self.imparam.imsize[1]
        )

        def exec_fn(l1, ltsv, overwrite_initial):
            return self._ellipsoid_exec(
                l1, ltsv, hp_scale, imageprefix, maxiter, eps, clean_box, nonnegative,
                resultasinitialimage, overwrite_initial, evaluator, ellipse_th, cos_th
            )
        return self._search_classical(l1_list, ltsv_list, exec_fn)

    def _ellipsoid_bayesian(self, l1_list, ltsv_list, hp_scale=1.0, imageprefix='image',
                             maxiter=1000, eps=1.0e-5, clean_box=None, nonnegative=True,
                             resultasinitialimage=True, bayesopt_maxiter=15,
                             ellipse_th=0.99, cos_th=0.99, bayesopt_n_startup_trials=None):
        """
        Select L1/Ltsv via Bayesian Optimization using the u-v-distance-grouped
        criterion (uvcriteria.UvEllipsoidEvaluator) instead of cross-validation.
        Each trial runs a single full-data MFISTA solve; no held-out visibility
        subsets are used, so this needs far fewer MFISTA solves than CV-based
        optimizers for a comparable number of BO trials.
        """
        evaluator = uvcriteria.UvEllipsoidEvaluator(
            self.working_set, self.imparam.imsize[0], self.imparam.imsize[1]
        )

        def exec_fn(l1, ltsv):
            return self._ellipsoid_exec(
                l1, ltsv, hp_scale, imageprefix, maxiter, eps, clean_box, nonnegative,
                resultasinitialimage, True, evaluator, ellipse_th, cos_th
            )
        return self._search_bayesian(l1_list, ltsv_list, exec_fn, bayesopt_maxiter,
                                     bayesopt_n_startup_trials)

    def _plot_cv_result(self, l1_list, ltsv_list, result, best_solution, figfile=None, optimizer='classical'):
        plotter_cls = None
        if optimizer == 'classical':
            plotter_cls = CVPlotter
        elif optimizer == 'bayesian':
            plotter_cls = CVBayesPlotter

        if not plotter_cls:
            return

        best_L1 = result.L1[best_solution]
        best_Ltsv = result.Ltsv[best_solution]
        best_mse = result.mse[best_solution]

        num_l1 = len(l1_list)
        num_ltsv = len(ltsv_list)

        plotter = plotter_cls(num_l1, num_ltsv, l1_list, ltsv_list)

        for mse, imagename, L1, Ltsv in zip(*result):
            imagearray = self.getimage(imagename)
            data = np.squeeze(imagearray.data)  # data will be 2D
            plotter.plotimage(L1, Ltsv, data, mse)

        if best_mse >= 0.0:
            plotter.mark_bestimage(best_L1, best_Ltsv)

        plotter.draw()
        if figfile is not None:
            plotter.savefig(figfile)

    def computemse(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None, nonnegative=True):
        """
        Compute mean-square-error (MSE) on resulting image.
        MSE is evaluated from visibility data provided as VisibilityWorkingSet
        instance.
        """
        mfistaparam = paramcontainer.MfistaParamContainer(l1=l1, ltsv=ltsv,
                                                          maxiter=maxiter, eps=eps,
                                                          clean_box=clean_box,
                                                          nonnegative=nonnegative)
        assert self.working_set is not None

        evaluator = cv.MeanSquareErrorEvaluator()
        num_fold = self.visset.num_fold

        if num_fold <= 1:
            # CV is disabled
            return -1.0

        subset_handler = cv.VisibilitySubsetHandler(self.visset)

        for subset in subset_handler.generate_subset(subset_id=0):

            # run MFISTA
            imagearray = self._solve(mfistaparam,
                                     subset.visibility_active,
                                     False, False)
            # evaluate MSE (Mean Square Error)
            mse = evaluator.evaluate_and_accumulate(subset.visibility_cache,
                                                    imagearray)

        mean_mse = evaluator.get_mean_mse()

        return mean_mse

    def computeapproximatemse(self):
        """
        Evaluate approximate mean-square-error (MSE) on resulting image.
        """
        raise NotImplementedError('Computation of Approximate MSE (LOOE) is not implemented yet.')
#         assert self.griddedvis is not None
#         evaluator = core.ApproximateCrossValidationEvaluator()
#
#         acv = evaluator.evaluate(self.griddedvis)
#         return 0.0

class CVPlotOuterFrame:
    def __init__(self, nv, nh, L1_list, Ltsv_list):
        self.nh = nh
        self.nv = nv

        self.left_margin = 0.1
        self.right_margin = 0.1
        self.bottom_margin = 0.1
        self.top_margin = 0.1
        total_width = 1.0 - (self.left_margin + self.right_margin)
        total_height = 1.0 - (self.bottom_margin + self.top_margin)
        dx = total_width / float(self.nh)
        dy = total_height / float(self.nv)
        self.dx = min(dx, dy)
        self.dy = self.dx
        f = plt.figure(num='CVPlot', figsize=(8, 8))
        plt.clf()
        left = self.left_margin
        bottom = self.bottom_margin
        height = self.dy * self.nv
        width = self.dx * self.nh
        outer_frame = plt.axes([left, bottom, width, height])
        outer_frame.set_xlim(-0.5, self.nh - 0.5)
        outer_frame.set_ylim(-0.5, self.nv - 0.5)
        outer_frame.set_xlabel('log10(Ltsv)')
        outer_frame.set_ylabel('log10(L1)')
        outer_frame.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(self.nh))))
        outer_frame.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(self.nv))))

        outer_frame.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: format_tick(x, Ltsv_list)))
        outer_frame.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: format_tick(x, L1_list)))

        self.axes = outer_frame
        self.figure = plt.gcf()


class CVPlotterBase:
    def __init__(self, nv, nh, L1_list, Ltsv_list):
        self.outer_frame = CVPlotOuterFrame(nv, nh, L1_list, Ltsv_list)

        self.L1_list = L1_list
        self.Ltsv_list = Ltsv_list

        self.image_height = self.outer_frame.dy
        self.image_width = self.outer_frame.dx

        self.axes_list = collections.defaultdict(dict)

    def plotimage(self, L1, Ltsv, data, mse):

        # Use np.isclose for tolerance-based comparisons
        # assert L1 in self.L1_list
        # assert Ltsv in self.Ltsv_list
        assert any(np.isclose(L1, val) for val in self.L1_list), f"L1: {L1} not found in L1_list"
        assert any(np.isclose(Ltsv, val) for val in self.Ltsv_list), f"Ltsv: {Ltsv} not found in Ltsv_list"

        # Find the index using np.isclose for tolerance-based matching
        # row = np.where(self.L1_list == L1)[0][0]
        # column = np.where(self.Ltsv_list == Ltsv)[0][0]
        row = np.where(np.isclose(self.L1_list, L1))[0][0]
        column = np.where(np.isclose(self.Ltsv_list, Ltsv))[0][0]

        cx = self.outer_frame.left_margin + (column + 0.5) * self.outer_frame.dx
        cy = self.outer_frame.bottom_margin + (row + 0.5) * self.outer_frame.dy
        left = cx - self.image_width / 2
        bottom = cy - self.image_height / 2
        #print 'plt.axes([{0}, {1}, {2}, {3}])'.format(left, bottom, width, height)
        nx, ny = data.shape
        a = plt.axes([left, bottom, self.image_width, self.image_height])
        a.imshow(np.flipud(data.transpose()))
        if mse >= 0.0:
            a.text(nx - 2, 5, '{:.5g}'.format(mse), ha='right', va='top', fontdict={'size': 'small', 'color': 'white'})
        a.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        a.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        self.axes_list[row][column] = a

    def mark_bestimage(self, L1, Ltsv):

        # Use np.isclose for tolerance-based comparisons
        # assert L1 in self.L1_list
        # assert Ltsv in self.Ltsv_list
        assert any(np.isclose(L1, val) for val in self.L1_list), f"L1: {L1} not found in L1_list"
        assert any(np.isclose(Ltsv, val) for val in self.Ltsv_list), f"Ltsv: {Ltsv} not found in Ltsv_list"

        # Find the index using np.isclose for tolerance-based matching
        # row = np.where(self.L1_list == L1)[0][0]
        # column = np.where(self.Ltsv_list == Ltsv)[0][0]
        row = np.where(np.isclose(self.L1_list, L1))[0][0]
        column = np.where(np.isclose(self.Ltsv_list, Ltsv))[0][0]

        best_axes = self.axes_list[row][column]
        bbox = best_axes.get_position()
        if int(matplotlib.__version__.split('.')[0]) > 1:
            best_frame = plt.axes(bbox, facecolor='none')
        else:
            best_frame = plt.axes(bbox, axisbg='none')
        best_frame.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        best_frame.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        for loc, spine in best_frame.spines.items():
            spine.set_color('red')
            spine.set_linewidth(3)

        axes_list = map(lambda x: x.values(), self.axes_list.values())
        max_zorder = max(map(lambda x: x.get_zorder(), itertools.chain(*axes_list)))
        best_axes.set_zorder(max_zorder + 1)
        plt.draw()

    def draw(self):
        plt.sca(self.outer_frame.axes)
        plt.draw()

    def savefig(self, figfile):
        plt.sca(self.outer_frame.axes)
        plt.savefig(figfile)


class CVPlotter(CVPlotterBase):
    pass


class CVBayesPlotter(CVPlotterBase):
    def __init__(self, nv, nh, L1_list, Ltsv_list):
        super().__init__(nv, nh, L1_list, Ltsv_list)

        n = max(self.outer_frame.nh, self.outer_frame.nv)
        if n > 9:
            self.image_height *= 2
            self.image_width *= 2
        elif n > 4:
            self.image_height *= 1.5
            self.image_width *= 1.5

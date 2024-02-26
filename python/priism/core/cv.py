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

import numpy as np
import scipy.interpolate
import contextlib

from . import datacontainer
from . import util


class VisibilitySubsetGenerator(object):
    def __init__(self, working_set, num_fold=10):
        self.working_set = working_set
        self.num_fold = num_fold

        # Visibility subset is meaningful only when num_fold > 1
        if self.num_fold > 1:
            # amplitude should be nonzero in active pixels
            self.num_active = len(self.working_set)
            self.active_index = range(self.num_active)
            print('num_active={0}'.format(self.num_active))

            # random index
            self.index_generator = util.RandomIndexGenerator(self.num_active, self.num_fold)
        else:
            self.active_index = None
            self.num_active = 0
            self.index_generator = None


class VisibilitySubsetHandler(object):
    def __init__(self, visset):
        # visset is VisibilitySubsetGenerator instance
        assert isinstance(visset, VisibilitySubsetGenerator)
        self.visibility = visset.working_set
        self.index_generator = visset.index_generator
        self.active_index = visset.active_index
        self.num_fold = visset.num_fold

        if self.num_fold <= 1:
            # Visibility subset generator is not properly configured
            raise RuntimeError('VisibilitySubsetGenerator is not properly configured. '
                               + 'Number of visibility subsets is less than 2 ({0})'.format(self.num_fold))

        self._clear()

    def _clear(self):
        self.visibility_active = None
        self.visibility_cache = None
        self.subset_id = None

        # amplitude should be nonzero in active pixels
        num_active = len(self.visibility)
        print('num_active={0}'.format(num_active))

    def generate_subset(self, subset_id):
        self.subset_id = subset_id

        # grid data shape: (nv, nu, npol, nchan)
        rdata = self.visibility.rdata
        idata = self.visibility.idata
        weight = self.visibility.weight
        u = self.visibility.u
        v = self.visibility.v

        for local_id in range(self.num_fold):
            self.subset_id = local_id

            # random index
            random_index = self.index_generator.get_subset_index(self.subset_id)
            #print('DEBUG_TN: subset ID {0} random_index = {1}'.format(self.subset_id, list(random_index)))

            # mask array for active visibility
            num_vis = len(self.visibility)
            mask = np.zeros(num_vis, dtype=bool)
            mask[:] = True
            mask[random_index] = False

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

            # visibility data to be cached
            # here, we assume npol == 1 (Stokes visibility I_v) and nchan == 1
            num_subvis = len(random_index)
            rcache = np.empty((num_subvis,), dtype=rdata.dtype)
            icache = np.empty_like(rcache)
            wcache = np.empty_like(rcache)
            ucache = np.empty((num_subvis,), dtype=u.dtype)
            vcache = np.empty_like(ucache)

            rcache[:] = rdata[random_index]
            icache[:] = idata[random_index]
            wcache[:] = weight[random_index]
            ucache[:] = u[random_index]
            vcache[:] = v[random_index]

            # generate subset
            assert num_subvis < num_vis
            num_active = num_vis - num_subvis
            ractive = np.empty((num_active,), dtype=rdata.dtype)
            iactive = np.empty_like(ractive)
            wactive = np.empty_like(ractive)
            uactive = np.empty((num_active,), dtype=u.dtype)
            vactive = np.empty_like(uactive)
            ractive[:] = rdata[mask]
            iactive[:] = idata[mask]
            wactive[:] = weight[mask]
            uactive[:] = u[mask]
            vactive[:] = v[mask]
            self.visibility_active = datacontainer.VisibilityWorkingSet(data_id=0,
                                                                        rdata=ractive,
                                                                        idata=iactive,
                                                                        weight=wactive,
                                                                        u=uactive,
                                                                        v=vactive)

            self.visibility_cache = [datacontainer.VisibilityWorkingSet(data_id=0,  # nominal data ID
                                                                        rdata=rcache,
                                                                        idata=icache,
                                                                        weight=wcache,
                                                                        u=ucache,
                                                                        v=vcache)]

            try:
                yield self
            finally:
                self._clear()


class GriddedVisibilitySubsetGenerator(object):
    def __init__(self, griddedvis, num_fold=10):
        self.griddedvis = griddedvis
        self.num_fold = num_fold

        # Visibility subset is meaningful only when num_fold > 1
        if self.num_fold > 1:
            # amplitude should be nonzero in active pixels
            grid_real = griddedvis.real
            grid_imag = griddedvis.imag
            self.active_index = np.where(np.logical_or(grid_real != 0, grid_imag != 0))
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
        # visset is GriddedVisibilitySubsetGenerator instance
        assert isinstance(visset, GriddedVisibilitySubsetGenerator)
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
        #self.active_index = np.where(np.logical_and(grid_real != 0, grid_imag != 0))
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
        #print('DEBUG_TN: subset ID {0} random_index = {1}'.format(self.subset_id, list(random_index)))

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
        u = np.empty((num_subvis,), dtype=np.float64)
        v = np.empty_like(u)
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
        assert gdata_shape[2] == 1  # npol should be 1
        assert gdata_shape[3] == 1  # nchan should be 1
        real = np.empty((num_subvis,), dtype=np.float32)
        imag = np.empty_like(real)
        wreal = np.empty_like(real)

        real[:] = grid_real[self.active_index][random_index]
        imag[:] = grid_imag[self.active_index][random_index]
        wreal[:] = wgrid_real[self.active_index][random_index]

        # generate subset
        self.visibility_active = self.visibility
        self.__replace_with(self.visibility_active.real, random_index, 0.0)
        self.__replace_with(self.visibility_active.imag, random_index, 0.0)
        self.__replace_with(self.visibility_active.wreal, random_index, 0.0)
        self.visibility_cache = [datacontainer.VisibilityWorkingSet(data_id=0,  # nominal data ID
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
        self.mse_storage = np.empty(100, dtype=np.float64)
        self.num_mse = 0

    def _evaluate_mse(self, visibility_cache, image):
        import time
        start_time = time.time()

        # Obtain visibility from image array
        shifted_image = np.fft.fftshift(image[:, :, 0, 0])
        shifted_imagefft = np.fft.fft2(shifted_image)
        imagefft = np.fft.ifftshift(shifted_imagefft)
        imagefft_transpose = imagefft.transpose()
        rmodel = imagefft_transpose.real
        imodel = imagefft_transpose.imag
        nx = rmodel.shape[0]
        ny = rmodel.shape[1]

        # Compute MSE
        mse = 0.0
        wsum = 0
        rinterp = scipy.interpolate.RectBivariateSpline(np.arange(nx), np.arange(ny), rmodel)
        iinterp = scipy.interpolate.RectBivariateSpline(np.arange(nx), np.arange(ny), imodel)
        for ws in visibility_cache:
            pu = ws.u
            pv = ws.v
            rdata = ws.rdata
            idata = ws.idata
            wdata = ws.weight
            adx = rdata - rinterp(pv, pu, grid=False)
            ady = idata - iinterp(pv, pu, grid=False)
            mse += np.sum(wdata * np.square(adx)) + np.sum(wdata * np.square(ady))
            wsum += np.sum(wdata)
        mse /= wsum
        end_time = time.time()
        print('Evaluate MSE: Elapsed {} sec'.format(end_time - start_time))
        return mse

    def evaluate_and_accumulate(self, visibility_cache, image):
        """
        Evaluate MSE (Mean Square Error) from image which is a solution of MFISTA
        and visibility_cache provided as a set of GridderWorkingSet instance.
        """
        # TODO: evaluate MSE
        mse = self._evaluate_mse(visibility_cache, image)

        # register it
        if self.num_mse >= len(self.mse_storage):
            self.mse_storage = np.resize(self.mse_storage, len(self.mse_storage) + 100)
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

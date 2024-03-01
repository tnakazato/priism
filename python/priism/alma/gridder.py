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

import os
from typing import Any, List, Tuple, TYPE_CHECKING

import numpy as np

import priism.core.util as util
import priism.core.datacontainer as datacontainer
from priism.external.casa import run_casa_task, CreateCasaQuantity, OpenMS, OpenTableForRead

if TYPE_CHECKING:
    from priism.alma.paramcontainer import GridParamContainer, ImageParamContainer, VisParamContainer

class GridderWorkingSet(datacontainer.VisibilityWorkingSet):
    """
    Working set for gridder

    NOTE: flag=True indicates *VALID* data
          flag=False indicates *INVALID* data

    data_id --- arbitrary data ID
    u, v --- position in uv-plane as pixel coordinate (nrow)
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
                 flag=None, weight=None, row_flag=None, channel_map=None,
                 pol_map=None):
        super(GridderWorkingSet, self).__init__(data_id, u, v, rdata, idata, weight)
        self.flag = flag
        self.row_flag = row_flag
        self.channel_map = channel_map
        self.pol_map = pol_map

    @property
    def pol_map(self):
        if hasattr(self, '_pol_map'):
            return self._pol_map
        else:
            self._pol_map = np.empty((self.npol,), dtype=np.int32)
            self._pol_map[:] = np.arange(self.npol)
            return self._pol_map

    @pol_map.setter
    def pol_map(self, value):
        if value is None:
            #self._pol_map = np.empty((self.npol,), dtype=np.int32)
            #self._pol_map[:] = range(self.npol)
            pass
        else:
            try:
                if len(value) == self.npol and isinstance(value[0], int):
                    self._pol_map = value
                else:
                    raise
            except Exception as e:
                raise ValueError('invalid pol_map ({0}). Should be int list or None.'.format(value))


# class GridFunctionUtil(object):
#     @staticmethod
#     def allocate(convsupport, convsampling, init=False):
#         n = (convsupport + 1) * convsampling * 2
#         gf = np.empty((n,), dtype=np.float32)
#         if init:
#             gf[:] = 0.0
#         return gf

#     @staticmethod
#     def box(convsupport, convsampling):
#         """
#         Generate Box gridding kernel whose value is 1.0 inside
#         convsupport pixel while 0.0 otherwise.

#         convsupport -- support radius in pixel
#         convsampling -- number of sampling per pixel
#         """
#         gf = GridFunctionUtil.allocate(convsupport, convsampling)
#         gf[:convsampling] = 1.0
#         gf[convsampling:] = 0.0
#         return gf

#     @staticmethod
#     def gauss(convsupport, convsampling, hwhm):
#         """
#         Generate Gaussian gridding kernel

#         convsupport -- support radius in pixel
#         convsampling -- number of sampling per pixel
#         hwhm -- Half-Width of Half-Maximum in pixel unit
#         """
#         gf = GridFunctionUtil.allocate(convsupport, convsampling)
#         gf[:] = 0.0
#         sigma = float(hwhm) / np.sqrt(2.0 * np.log(2.0))
#         m = convsupport * convsampling
#         for i in range(m):
#             x = float(i) / float(convsampling)
#             gf[i] = np.exp(-(x * x) / (2.0 * sigma * sigma))
#         return gf

#     @staticmethod
#     def sf(convsupport, convsampling):
#         """
#         Generate prolate-Spheroidal gridding kernel

#         convsupport -- support radius in pixel
#         convsampling -- number of sampling per pixel
#         """
#         gf = GridFunctionUtil.allocate(convsupport, convsampling)
#         m = convsupport * convsampling
#         for i in range(m):
#             nu = float(i) / float(m)
#             val = GridFunctionUtil.grdsf(nu)
#             gf[i] = (1.0 - nu * nu) * val
#         gf[m:] = 0.0
#         # normalize so peak is 1.0
#         gf *= 1.0 / gf[0]
#         return gf

#     @staticmethod
#     def grdsf(nu):
#         """
#         cf. casacore/scimath_f/grdsf.f
#         """
#         P0 = [8.203343e-2, -3.644705e-1, 6.278660e-1,
#               -5.335581e-1, 2.312756e-1]
#         P1 = [4.028559e-3, -3.697768e-2, 1.021332e-1,
#               -1.201436e-1, 6.412774e-2]
#         Q0 = [1.0000000e0, 8.212018e-1, 2.078043e-1]
#         Q1 = [1.0000000e0, 9.599102e-1, 2.918724e-1]
#         nP = 4
#         nQ = 2

#         val = 0.0
#         if 0.0 <= nu and nu < 0.75:
#             P = P0
#             Q = Q0
#             nuend = 0.75
#         elif 0.75 <= nu and nu <= 1.0:
#             P = P1
#             Q = Q1
#             nuend = 1.0
#         else:
#             val = 0.0
#             return val

#         top = P[0]
#         delnusq = nu * nu - nuend * nuend
#         kdelnusq = 1.0
#         for k in range(1, nP + 1):
#             kdelnusq *= delnusq
#             top += P[k] * kdelnusq

#         bot = Q[0]
#         kdelnusq = 1.0
#         for k in range(1, nQ + 1):
#             kdelnusq *= delnusq
#             bot += Q[k] * kdelnusq

#         if bot != 0.0:
#             val = top / bot
#         else:
#             val = 0.0

#         return val


class GridderResult(object):
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


def is_channel_value(value: Any) -> bool:
    """Check if given quantity string doesn't have unit.

    Args:
        value: string to be inspected

    Returns:
        True if it doesn't have unit, False otherwise.
    """
    if isinstance(value, str):
        return value.isdigit()

    return isinstance(value, int)



class VisibilityGridder(object):
    """
    Configure grid and accumulate data onto each grid position.
    Data should be provided in the form of GridderWorkingSet
    instance.
    """
    GRID_DATA_DIR = '.priism'
    GRID_DATA_NAME = 'gridder.ms'

    def __init__(
            self, gridparam: 'GridParamContainer',
            visparams: 'List[VisParamContainer]',
            imageparam: 'ImageParamContainer'
        ):
        """Initialize gridder.

        Args:
            gridparam: Gridding parameter
            visparams: List of visibility parameters
            imageparam: Imaging parameter
        """
        self.gridparam = gridparam
        self.visparams = visparams
        self.imageparam = imageparam
        # self.num_ws = 0
        self._init()

    def _init(self):
        os.makedirs(self.GRID_DATA_DIR, exist_ok=True)

    def _init_old(self):

        # grid parameter from visibility selection parameter
        #  sel = self.visparam.as_msindex()
        #  poldd = sel['poldd']
        #  with casa.OpenTableForRead(os.path.join(self.visparam.vis,
        #                                          'DATA_DESCRIPTION')) as tb:
        #      tsel = tb.selectrows(poldd)
        #      polids = np.unique(tsel.getcol('POLARIZATION_ID'))
        #      tsel.close()
        #  with casa.OpenTableForRead(os.path.join(self.visparam.vis,
        #                                          'POLARIZATION')) as tb:
        #      tsel = tb.selectrows(polids)
        #      num_corrs = np.unique(tsel.getcol('NUM_CORR'))
        #      tsel.close()

        #  assert len(num_corrs) == 1
        #  self.npol = num_corrs[0]
        # so far npol should be 1 (only I)
        self.npol = 1

        # grid parameter from image parameter
        uvgridconfig = self.imageparam.uvgridconfig
        self.cellu = uvgridconfig.cellu
        self.cellv = uvgridconfig.cellv
        self.nu = uvgridconfig.nu
        self.nv = uvgridconfig.nv
        self.offsetu = uvgridconfig.offsetu
        self.offsetv = uvgridconfig.offsetv

        # create storage
        # require margin based on convsupport parameter since
        # sakura gridding function ignore convsupport-pixels
        # from spatial edges
        wsshape = (self.npol, self.nchan)
        self.wsum_real = np.empty(wsshape, dtype=np.float64)
        self.wsum_imag = np.empty(wsshape, dtype=np.float64)
        gridshape = (self.nv + 2 * self.convsupport,
                     self.nu + 2 * self.convsupport,
                     self.npol, self.nchan)
        self.grid_real = np.empty(gridshape, dtype=np.float32)
        self.grid_imag = np.empty(gridshape, dtype=np.float32)
        self.wgrid_real = np.empty(gridshape, dtype=np.float32)
        self.wgrid_imag = np.empty(gridshape, dtype=np.float32)

        # zero clear
        self._clear_grid()

        # number of ws to be gridded
        self.num_ws = 0

    def _clear_grid(self):
        self.grid_real[:] = 0
        self.grid_imag[:] = 0
        self.wgrid_real[:] = 0
        self.wgrid_imag[:] = 0
        self.wsum_real[:] = 0
        self.wsum_imag[:] = 0

    def __to_freq(self, visparam: 'VisParamContainer', start: str, width: str) -> Tuple[str, str]:
        """Convert channel value into frequency quantity.

        Args:
            visparam: Visibility data selection parameter
            start: start value of image's spectral axis
            width: width of image's spectral axis

        Returns:
            start and width as frequency quantities
        """
        vis = visparam.vis
        with OpenMS(vis) as ms:
            ms.msselect({'spw': visparam.spw})
            selected = ms.msselectedindices()
            selected_spws = selected['spw']

        num_spws = len(selected_spws)
        with OpenTableForRead(os.path.join(vis, 'SPECTRAL_WINDOW')) as tb:
            if is_channel_value(start):
                if num_spws > 1:
                    raise RuntimeError('Spw selection must be unique when start is specified as channel.')
                spw_id = selected_spws[0]
                chan_freq = tb.getcell('CHAN_FREQ', spw_id)
                chan_width = tb.getcell('CHAN_WIDTH', spw_id)
                nchan = len(chan_freq)
                start = int(start)
                if 0 <= start and start < nchan:
                    start = chan_freq[start] - chan_width[start] / 2
                else:
                    freq_0 = chan_freq[0] - chan_width[0] / 2
                    start = freq_0 + start * chan_width[0]

            if is_channel_value(width):
                if num_spws > 1:
                    raise RuntimeError('Spw selection must be unique when width is specified as channel.')
                spw_id = selected_spws[0]
                width = int(width)
                chan_width = tb.getcell('CHAN_WIDTH', spw_id)
                width = width * chan_width[0]

        return f'{start}Hz', f'{width}Hz'

    def _configure_spectral_axis(self, visparam: 'VisParamContainer') -> dict:
        """Configure image channel for given MS.

        Args:
            visparam: Visibility data selection parameter

        Returns:
            image channel configuration as dict
        """
        start = self.imageparam.start
        width = self.imageparam.width
        nchan = self.imageparam.nchan
        start, width = self.__to_freq(visparam, start, width)

        return {
            'start': start,
            'width': width,
            'nchan': nchan
        }

    def grid(self):
        """
        Accumulate data provided as a list of working set onto grid.
        """
        # for ws in ws_list:
        #     self.grid_ws(ws)
        params = {}
        # configure msuvbin parameters here
        params['outputvis'] = os.path.join(
            self.GRID_DATA_DIR,
            self.GRID_DATA_NAME
        )
        params['imsize'] = self.imageparam.imsize
        params['cell'] = self.imageparam.cell
        if not isinstance(params['cell'], str):
            # could be a list
            params['cell'] = params['cell'][0]
        params['ncorr'] = 1  # Stokes I
        params['wproject'] = self.gridparam.wproject
        params['memfrac'] = self.gridparam.memfrac
        for visparam in self.visparams:
            params['vis'] = visparam.vis
            # data selection parameters
            params.update(
                visparam.as_msselection()
            )
            params.update(
                self._configure_spectral_axis(visparam)
            )
            run_casa_task('msuvbin', **params)

        return self._get_ws()

    def _get_ws(self):
        vis = os.path.join(self.GRID_DATA_DIR, self.GRID_DATA_NAME)
        with OpenTableForRead(vis) as tb:
            data = tb.getcol('DATA')
            weight_sp = tb.getcol('WEIGHT_SPECTRUM')
            nrow = tb.nrows()

        # (u, v) is converted to array indices.
        # Data order is assumed to be as follows:
        #
        #        0:(u0, v0),        1:(u1, v0), ...,  N-1:(uN, v0),
        #        N:(u0, v1),      N+1:(u1, v1), ..., 2N-1:(uN, v1),
        #   ...
        #   (M-1)N:(u0, vM), (M-1)N+1:(u1, vM), ..., MN-1:(uN, vM)
        #
        # where the format is array_index:(u_vaue, v_value).
        nu = self.imageparam.imsize[0]
        nv = self.imageparam.imsize[1]
        offsetu = self.imageparam.uvgridconfig.offsetu
        offsetv = self.imageparam.uvgridconfig.offsetv
        assert nrow == nu * nv

        u_all = np.empty(nrow, dtype=int)
        v_all = np.empty_like(u_all)
        for i in range(nv):
            start = i * nu
            end = start + nu
            v_all[start:end] = nv - i - 1
            if i == 0:
                u_all[start:end] = np.arange(nu, dtype=int)
            else:
                u_all[start:end] = u_all[start - nu:start]

        idx = weight_sp[0, 0].nonzero()[0]
        ndata = len(idx)

        # we should hold visibilities read from MS as well as
        # their complex conjugates
        u = np.empty(2 * ndata, dtype=int)
        v = np.empty_like(u)
        rdata = np.empty_like(u, dtype=complex)
        idata = np.empty_like(rdata)
        weight = np.empty_like(u, dtype=float)

        # data from gridded MS
        u[:ndata] = u_all[idx]
        v[:ndata] = v_all[idx]
        data_nonzero = data[0, 0, idx]
        rdata[:ndata] = data_nonzero.real
        idata[:ndata] = data_nonzero.imag
        weight[:ndata] = weight_sp[0, 0, idx]

        # complex conjugates
        u[ndata:] = 2 * offsetu - u[:ndata]
        v[ndata:] = 2 * offsetv - v[:ndata]
        rdata[ndata:] = rdata[:ndata]
        idata[ndata:] = - idata[:ndata]
        weight[ndata:] = weight[:ndata]

        return datacontainer.VisibilityWorkingSet(
            data_id=0,
            u=u,
            v=v,
            rdata=rdata,
            idata=idata,
            weight=weight
        )

    def grid_ws(self, ws):
        """
        Accumulate data provided as working set onto grid.
        """
        #print('LOG: accumulate visibility chunk #{0} onto grid'.format(ws.data_id))
        # shift uv (pixel) coordinates by convsupport to take into
        # account margin pixels
        ws.u += self.convsupport
        ws.v += self.convsupport
        tmpu = ws.u.copy()
        tmpv = ws.v.copy()
        tmpd = ws.idata.copy()
        try:
            # grid ws as it is
            sakura.grid(ws, self.gridfunction,
                        self.convsupport, self.convsampling,
                        self.weight_only,
                        self.grid_real, self.grid_imag,
                        self.wgrid_real, self.wgrid_imag,
                        self.wsum_real, self.wsum_imag)

            # then grid complex conjugate of ws
            ws.u[:] = 2 * self.offsetu - (ws.u - self.convsupport) + self.convsupport
            ws.v[:] = 2 * self.offsetv - (ws.v - self.convsupport) + self.convsupport
            ws.idata *= -1.0
            sakura.grid(ws, self.gridfunction,
                        self.convsupport, self.convsampling,
                        self.weight_only,
                        self.grid_real, self.grid_imag,
                        self.wgrid_real, self.wgrid_imag,
                        self.wsum_real, self.wsum_imag)
        finally:
            # restore orignal uv (pixel) coordinates
            ws.u[:] = tmpu
            ws.v[:] = tmpv
            ws.idata[:] = tmpd
            ws.u -= self.convsupport
            ws.v -= self.convsupport
            self.num_ws += 1

    def get_result(self):
        # remove margin from grid array
        grid_shape = self.grid_real.shape
        outgrid_shape = (grid_shape[0] - 2 * self.convsupport,
                         grid_shape[1] - 2 * self.convsupport,
                         grid_shape[2], grid_shape[3])
        grid_real = self.grid_real[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        grid_imag = self.grid_imag[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        wgrid_real = self.wgrid_real[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        wgrid_imag = self.wgrid_imag[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        assert grid_real.shape == outgrid_shape
        assert grid_imag.shape == outgrid_shape
        assert wgrid_real.shape == outgrid_shape
        assert wgrid_imag.shape == outgrid_shape

        nonzero_real = np.where(wgrid_real != 0.0)
        nonzero_imag = np.where(wgrid_imag != 0.0)
        uvgrid_real = np.empty(outgrid_shape, dtype=np.float64)
        uvgrid_imag = np.empty(outgrid_shape, dtype=np.float64)
        uvgrid_wreal = np.empty(outgrid_shape, dtype=np.float64)
        uvgrid_wreal[:] = wgrid_real
        if np.all(wgrid_real == wgrid_imag):
            uvgrid_wimag = None
        else:
            uvgrid_wimag = np.empty(outgrid_shape, dtype=np.float64)
            uvgrid_wimag[:] = wgrid_imag
        uvgrid_real[:] = 0.0
        uvgrid_imag[:] = 0.0
        uvgrid_real[nonzero_real] = grid_real[nonzero_real] / wgrid_real[nonzero_real]
        uvgrid_imag[nonzero_imag] = grid_imag[nonzero_imag] / wgrid_imag[nonzero_imag]

        result = datacontainer.GriddedVisibilityStorage(uvgrid_real, uvgrid_imag,
                                                        uvgrid_wreal, uvgrid_wimag,
                                                        self.num_ws)
        return result

    def get_result2(self):
        # remove margin from grid array
        grid_shape = self.grid_real.shape
        outgrid_shape = (grid_shape[0] - 2 * self.convsupport,
                         grid_shape[1] - 2 * self.convsupport,
                         grid_shape[2], grid_shape[3])
        grid_real = self.grid_real[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        grid_imag = self.grid_imag[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        wgrid_real = self.wgrid_real[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        wgrid_imag = self.wgrid_imag[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        assert grid_real.shape == outgrid_shape
        assert grid_imag.shape == outgrid_shape
        assert wgrid_real.shape == outgrid_shape
        assert wgrid_imag.shape == outgrid_shape

        ws = datacontainer.grid2ws(grid_real, grid_imag, wgrid_real, wgrid_imag)

        # normalize visibility data
        ws.rdata /= ws.weight
        ws.idata /= ws.weight

        return ws


class CrossValidationVisibilityGridder(VisibilityGridder):
    """
    Gridder for cross validation.
    To separate visibility subset for cross validation,
    this class has cache mechanism that separate a certain
    amount of raw visibility from gridding and store them
    for cross validation.
    """
    def __init__(self, gridparam, imageparam, num_ws, num_fold=10):
        super(CrossValidationVisibilityGridder, self).__init__(gridparam, imageparam)
        self.num_ws = num_ws
        self.num_fold = num_fold

        self.visibility_cache = []

        self.index_generator = util.VisibilitySubsetGenerator(self.num_ws, self.num_fold)

    def grid(self, ws, subset_id):
        """
        Separate ws if it is judged as a visibility cache for cross validation.
        Otherwise, accumulate ws onto grid.
        """
        subset_index = self.index_generator.get_subset_index(subset_id)
        #if ws.data_id % num_fold == subset_id:
        if ws.data_id in subset_index:
            self.visibility_cache.append(ws)
        else:
            super(CrossValidationVisibilityGridder).grid(ws)

    def get_visibility_cache(self):
        return self.visibility_cache

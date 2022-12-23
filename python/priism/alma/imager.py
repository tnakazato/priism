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

from . import paramcontainer
from . import gridder
from . import visreader
from . import visconverter
from . import imagewriter
import priism.external.sakura as sakura
import priism.external.casa as casa

import priism.core.imager as core_imager
import priism.core.datacontainer as datacontainer


class AlmaSparseModelingResult(object):
    """
    This is a class to hold a result produced by AlmaSparseModeling
    """
    def __init__(self, imagename, cv=-1.0, acv=-1.0):
        """
        Constructor

        Parameters:
            imagename  name of the FITS cube
            cv         associating cross validation
            acv        associating approximate cross validation
        """
        self.imagename = imagename
        self.cv = cv
        self.acv = acv

    def __repr__(self):
        return 'imagename "{0}"\n'.format(self.imagename) \
               + '  cross validation           = {0}\n'.format(self.cv) \
               + '  cross validation (approx.) = {0}\n'.format(self.acv)


class AlmaSparseModelingImager(core_imager.SparseModelingImager):
    """
    AlmaSparseModelingImager inherits all core functions from its parent.
    It performs visibility gridding on uv-plane.
    It additionally equips to compute direct and approximate cross
    validation of resulting image as well as a function to export
    resulting image as an FITS cube.
    """
    @property
    def imagesuffix(self):
        """
        Image product of AlmaSparseModelingImage is FITS file.
        Therefore, a suffix for image product should be 'fits'.
        """
        return 'fits'

#     """
#     Core implementation of sparse modeling specialized for ALMA.
#     It performs visibility gridding on uv-plane.
#     """
    def __init__(self, solver='mfista_fft'):
        """
        Constructor

        Parameters:
            solver  name of the solver
                    choices are as follows.
                      'mfista_fft'    MFISTA algorithm with FFT by S. Ikeda.
                      'mfista_nufft'  MFISTA algorithm with NUFFT by S. Ikeda
        """
        super(AlmaSparseModelingImager, self).__init__(solver)

    def selectdata(self, vis, field='', spw='', timerange='', uvrange='', antenna='',
                   scan='', observation='', intent='', datacolumn='corrected'):
        """
        Select visibility data.

        Parameters:
            vis             name of measurement set
            field           field selection (default '' ---> all)
            spw             spw selection (default '' ---> all)
            timerange       timerange selection (default '' ---> all)
            uvrange         uvrange selectoin  (default '' ---> all)
            antenna         antenna/baseline selection (default '' ---> all)
            scan            scan selection (default '' ---> all)
            observation     observation ID selection (default '' ---> all)
            intent          intent selection (default '' ---> all)
            datacolumn      data column to be used ('data' or 'corrected' or 'residual')
        """
        visparam = paramcontainer.VisParamContainer.CreateContainer(**locals())
        self.visparams.append(visparam)

    def defineimage(self, imsize=100, cell='1arcsec', phasecenter='', projection='SIN',
                    nchan=-1, start='', width='', outframe='LSRK', stokes='I'):
        """
        Define resulting image.

        start, width, and nchan are defined as follows:

          start=<center frequency of first image channel>
            |
        |-------|-------|-------| nchan=3
        |<----->|
          width=<constant channel width of image channel>


        Parameters:
            imsize          number of pixels for the resulting image
                            (default 100 ---> [100,100])
            cell            pixel size for the resulting image
                            (default '1arcsec' ---> ['1arcsec', '1arcsec']
            phasecenter     phase center direction or field ID (default '')
            projection      projection scheme (default 'SIN')
            nchan           number of spectral channels
            start           start channel/frequency
            width           width in channel/frequency
            outframe        output frequency reference frame (fixed to 'LSRK')
            stokes          stokes parameter (fixed to 'I')
        """
        self.imparam = paramcontainer.ImageParamContainer.CreateContainer(**locals())

    def configuregrid(self, convsupport, convsampling, gridfunction):
        if isinstance(gridfunction, str):
            gridfunction = gridder.GridFunctionUtil.sf(convsupport, convsampling)
        self.gridparam = paramcontainer.GridParamContainer.CreateContainer(**locals())

    @casa.adjust_casalog_level('WARN')
    def gridvis(self, parallel=False):
        """
        Grid visibility data on uv-plane.
        """
        # gridvis consists of several steps:
        #     1. select and read data according to data selection
        #     2. pre-gridding data processing
        #     3. give the data to gridder
        #     4. post-gridding data processing
        #
        visgridder = gridder.VisibilityGridder(self.gridparam, self.imparam)

        # workaround for strange behavior of ms iterator
        interval = 1.0e-16
        for visparam in self.visparams:
            reader = visreader.VisibilityReader(visparam)
            converter = visconverter.VisibilityConverter(visparam, self.imparam)
            if parallel:
                for working_set in sakura.paraMap(self.num_threads,
                                                  converter.generate_working_set,
                                                  reader.readvis(interval=interval)):
                    visgridder.grid(working_set)
            else:
                for chunk in reader.readvis(interval=interval):
                    working_set = converter.generate_working_set(chunk)
                    visgridder.grid(working_set)
        self.griddedvis = visgridder.get_result()
        self.working_set = visgridder.get_result2()

    @casa.adjust_casalog_level('WARN')
    def readvis(self, parallel=False):
        """
        Read visibility data
        """
        u = []
        v = []
        real = []
        imag = []
        weight = []
        interval = 1.0e-16
        for visparam in self.visparams:
            reader = visreader.VisibilityReader(visparam)
            converter = visconverter.VisibilityConverter(visparam, self.imparam)
            if parallel:
                raise NotImplementedError()
            else:
                for chunk in reader.readvis(interval=interval):
                    ws_list = converter.generate_working_set(chunk)
                    for ws in ws_list:
                        flag = ws.flag
                        valid = np.where(flag == True)
                        u.extend(ws.u[valid[0]])
                        v.extend(ws.v[valid[0]])
                        real.extend(ws.rdata[valid])
                        imag.extend(ws.idata[valid])
                        weight.extend(ws.weight[(valid[0], valid[2])])

        self.working_set = datacontainer.VisibilityWorkingSet(data_id=0,
                                                              u=np.asarray(u),
                                                              v=np.asarray(v),
                                                              rdata=np.asarray(real, dtype=np.float64),
                                                              idata=np.asarray(imag, dtype=np.float64),
                                                              weight=np.asarray(weight, dtype=np.float64))

    def exportimage(self, imagename, overwrite=False):
        """
        Export MFISTA result as an image (FITS cube).
        mfista must be executed beforehand.

        Parameters:
            imagename  name of output image name
        """
        if self.imparam is None:
            raise RuntimeError('You have to define image configuration before export!')
        self.imparam.imagename = imagename

        if self.imagearray is None:
            raise RuntimeError('You don\'t have an image array!')

        # convert phasecenter if it is given as FIELD_ID
        vis = self.visparams[0].vis
        if isinstance(self.imparam.phasecenter, str) and self.imparam.phasecenter.isdigit():
            print('Use PHASE_DIR for FIELD {0}'.format(self.imparam.phasecenter))
            # take first MS
            field_id = int(self.imparam.phasecenter)
            phase_direction = imagewriter.ImageWriter.phase_direction_for_field(vis=vis,
                                                                                field_id=field_id)
            self.imparam.phasecenter = phase_direction
        if (isinstance(self.imparam.start, str) and self.imparam.start.isdigit()) \
           or isinstance(self.imparam.start, int):
            # TODO: we need LSRK frequency
            start = self.imparam.start
            spw = int(self.visparams[0].as_msindex()['spw'][0])
            print('Use Freuquency for channel {0} spw {1}'.format(start, spw))
            cf, cw = imagewriter.ImageWriter.frequency_setup_for_spw(vis=vis,
                                                                     spw_id=spw,
                                                                     chan=start)
            self.imparam.start = cf
            self.imparam.width = cw
        imagemeta = paramcontainer.ImageMetaInfoContainer.fromvis(vis)
        writer = imagewriter.ImageWriter(self.imparam, self.imagearray.data,
                                         imagemeta)
        writer.write(overwrite=overwrite)

    def getimage(self, imagename):
        with casa.OpenImage(imagename) as ia:
            chunk = ia.getchunk()
        return datacontainer.ResultingImageStorage(chunk)

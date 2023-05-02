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
import os

import priism.external.casa as casa
from . import paramcontainer


class ImageWriter(object):
    """
    Create an FITS cube from given image array and coordinate information.
    """
    @staticmethod
    def phase_direction_for_field(vis, field_id):
        with casa.OpenMSMetaData(vis) as msmd:
            pdir = msmd.phasecenter(field_id)
        return pdir

    @staticmethod
    def frequency_setup_for_spw(vis, spw_id, chan):
        with casa.OpenTableForRead(os.path.join(vis, 'SPECTRAL_WINDOW')) as tb:
            chan_freq = tb.getcell('CHAN_FREQ', spw_id)
            chan_width = tb.getcell('CHAN_WIDTH', spw_id)
        return '{0:16.12f}Hz'.format(chan_freq[chan]), '{0:16.12f}Hz'.format(chan_width[chan])

    def __init__(self, imageparam, imagearray, imagemeta=None):
        self.imageparam = imageparam
        self.imagearray = imagearray
        if imagemeta is None:
            self.imagemeta = paramcontainer.ImageMetaInfoContainer()
        else:
            self.imagemeta = imagemeta

    def write(self, overwrite=False):
        ia = casa.CreateCasaImageAnalysis()
        # configure coordinate system
        csys = self._setup_coordsys()

        # image array reshape
        imgshape = self.imagearray.shape
        if len(imgshape) == 2:
            # (nx,ny)
            arr = self.imagearray.reshape((imgshape[0], imgshape[1], 1, 1))
        elif len(imgshape) == 3:
            # (nx,ny,nchan)
            arr = self.imagearray.reshape((imgshape[0], imgshape[1], 1, imgshape[2]))
        elif len(imgshape) == 4:
            # (nx,ny,nstokes,nchan)
            arr = self.imagearray
        else:
            raise ValueError('image array is not correct shape')

        ia.fromarray(pixels=arr, csys=csys.torecord())
        ia.setbrightnessunit('Jy/pixel')
        status = ia.tofits(outfile=self.imageparam.imagename, optical=False,
                           stokeslast=True, overwrite=overwrite)
        ia.done()

        return status

    def _setup_coordsys(self):
        csys = casa.CreateCasaCoordSys()
        me = casa.CreateCasaMeasure()
        qa = casa.CreateCasaQuantity()

        c = csys.newcoordsys(direction=True, spectral=True, stokes=self.imageparam.stokes)

        # direction coordinate
        phasecenter = self.imageparam.phasecenter
        print('DEBUG phasecenter={0}'.format(phasecenter))
        refframe = me.getref(phasecenter)
        refpix = [int(x) // 2 for x in self.imageparam.imsize]
        center = me.getvalue(phasecenter)
        refval = [center['m0']['value'], center['m1']['value']]
        q2s = lambda x: '{0} {1}'.format(x['value'], x['unit'])
        srefval = list(map(q2s, [center['m0'], center['m1']]))
        incr = list(map(qa.quantity, self.imageparam.cell))
        # increment of horizontal axis should be negative
        incr[0] = qa.mul(-1.0, incr[0])
        sincr = list(map(q2s, incr))
        projection = self.imageparam.projection
        print('DEBUG refpix={0}, refval={1}'.format(refpix, refval))
        c.setdirection(refcode=refframe,
                       proj=projection,
                       refpix=refpix,
                       refval=srefval,
                       incr=sincr)

        # spectral coordinate
        refframe = 'LSRK'
        print('start {0} width {1}'.format(self.imageparam.start,
                                           self.imageparam.width))
        start = qa.convert(self.imageparam.start, 'Hz')
        width = qa.convert(self.imageparam.width, 'Hz')
        nchan = self.imageparam.nchan
        f = np.fromiter((start['value'] + i * width['value'] for i in range(nchan)), dtype=np.float64)
        print('f = {0}'.format(f))
        frequencies = qa.quantity(f, 'Hz')
        veldef = 'radio'
        if len(f) > 1:
            c.setspectral(refcode=refframe,
                          frequencies=frequencies,
                          doppler=veldef)
        else:
            print('set increment for spectral axis: {0}'.format(width))
            #c.setreferencepixel(value=0, type='spectral')
            #c.setreferencevalue(value=start, type='spectral')
            #c.setincrement(value=width, type='spectral')
            r = c.torecord()
            if 'spectral2' in r:
                key = 'spectral2'
            elif 'spectral1' in r:
                key = 'spectral1'
            r[key]['wcs']['crpix'] = 0.0
            r[key]['wcs']['crval'] = start['value']
            r[key]['wcs']['cdelt'] = width['value']
            c.fromrecord(r)

        # Stokes coordinate
        # currently only 'I' image is supported
        #c.setstokes('I')
        #c.setincrement(value=1, type='stokes')

        # Meta info
        c.setobserver(self.imagemeta.observer)
        c.settelescope(self.imagemeta.telescope)
        c.setepoch(self.imagemeta.observing_date)
        rest_frequency = self.imagemeta.rest_frequency
        if rest_frequency is None or (isinstance(rest_frequency, str) and len(rest_frequency) == 0):
            f = frequencies['value']
            nchan = len(f)
            if nchan % 2 == 0:
                c1 = (nchan - 1) // 2
                c2 = c1 + 1
                rest_frequency = qa.quantity(0.5 * (f[c1] + f[c2]), frequencies['unit'])
            else:
                c1 = (nchan - 1) // 2
                rest_frequency = qa.quantity(f[c1], frequencies['unit'])

        print('rest_frequency={0}'.format(rest_frequency))
        if qa.checkfreq(rest_frequency) and qa.gt(rest_frequency, qa.quantity(0, 'Hz')):
            c.setrestfrequency(rest_frequency)

        print(c.summary(list=False)[0])

        return c


def parse_phasecenter(phasecenter_str):
    qa = casa.CreateCasaQuantity()
    me = casa.CreateCasaMeasure()

    if len(phasecenter_str) == 0:
        # defualt value '0:0:0 0.0.0 J2000'
        lat = qa.quantity(0.0, 'rad')
        lon = qa.quantity(0.0, 'rad')
        ref = 'J2000'
    else:
        # expected format: "longitude latitude [ref]"
        s = phasecenter_str.split()
        ref = 'J2000'  # default reference is J2000
        if len(s) == 3:
            lat = qa.quantity(s[1])
            lon = qa.quantity(s[0])
            ref = s[2]
        elif len(s) == 2:
            lat = qa.quantity(s[1])
            lon = qa.quantity(s[0])
        else:
            raise ValueError('Invalid phasecenter: "{0}"'.format(phasecenter_str))

    #print 'DEBUG rf={0} lon={1} lat={2}'.format(ref, lon, lat)
    direction = me.direction(rf=ref, v0=lon, v1=lat)
    return direction

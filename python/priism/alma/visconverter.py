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
import numpy as np
import scipy.interpolate as interpolate
import re

import priism.external.casa as casa
import priism.external.sakura as sakura
from . import gridder


class VisibilityConverter(object):
    """
    VisibilityConverter implements a conversion between
    raw visibility data to working set for gridder.
    """
    required_columns = ['time', 'uvw', 'field_id', 'data_desc_id',
                        'data', 'flag', 'weight']
    frequency_reference = ['REST',
                           'LSRK',
                           'LSRD',
                           'BARY',
                           'GEO',
                           'TOPO',
                           'GALACTO',
                           'LGROUP',
                           'CMB']

    def __init__(self, visparam, imageparam):
        self.visparam = visparam
        self.imageparam = imageparam

        self._warn_refocus()

        self.inspect_data()

    def _to_stokesI(self, data_in, flag_in, weight_in, weight_factor,
                    real_out, imag_out, flag_out, weight_out):
        """
        Convert XXYY (or RRLL) correlations into Stokes Visibility I_v.
        Formula for conversion is,

            I_v = (XX + YY) / 2
            W_I = 4 W_XX W_YY / (W_XX + W_YY)

        NOTE:
           shape of input data are (npol, nchan, nrow)
           shape of output data are (nrow, npol, nchan)
        """
        npol = data_in.shape[0]
        nrow = data_in.shape[2]

        if npol == 1:
            # single polarization, just copy
            real_out[:] = data_in.real.transpose((2, 0, 1))
            imag_out[:] = data_in.imag.transpose((2, 0, 1))
            flag_out[:] = flag_in.transpose((2, 0, 1))
            for irow in range(nrow):
                weight_out[irow] = weight_in[irow] * weight_factor
            return
        elif npol == 4:
            # data might be full Stokes parameters (IQUV)
            # pick up Stokes I (first polarization component)
            # TODO: differentiate Stokes parameter and four correlations (XX,XY,YX,YY)
            real_out[:] = data_in.real.transpose((2, 0, 1))[:, :1, :]
            imag_out[:] = data_in.imag.transpose((2, 0, 1))[:, :1, :]
            flag_out[:] = flag_in.transpose((2, 0, 1))[:, :1, :]
            #print weight_in.shape
            for irow in range(nrow):
                weight_out[irow] = weight_in[0, irow] * weight_factor
            return

        # here npol should be 2 (dual polarization XXYY or RRLL)
        assert npol == 2

        # data
        mask = np.where(flag_in == False, 0.5, 0.0)
        data_out = (data_in * mask).sum(axis=0)
        real_out[:, 0, :] = data_out.real.transpose((1, 0))
        imag_out[:, 0, :] = data_out.imag.transpose((1, 0))
        del mask
        del data_out

        # flag
        flag_out[:] = True
        for ipol in range(npol):
            flag_out[:] = np.logical_and(flag_out,
                                            flag_in[ipol:ipol + 1, :, :].transpose((2, 0, 1)))

        # weight
        # weight_in.shape = (npol, nrow)
        # weight_out.shape = (nrow, nchan)
        for irow in range(nrow):
            w1 = weight_in[0, irow] * weight_factor
            w2 = weight_in[1, irow] * weight_factor
            weight_out[irow] = 4.0 * w1 * w2 / (w1 + w2)

    def freq_ref_string(self, type_id):
        if type_id < 0 or len(self.frequency_reference) <= type_id:
            return 'UNDEFINED'
        else:
            return self.frequency_reference[type_id]

    def inspect_data(self):
        """
        Inspect data given as a field 'vis' in visparam.
        """
        vis = self.visparam.vis

        # make mapping between DATA_DESC_ID and SPECTRAL_WINDOW_ID/POLARIZATION_ID
        with casa.OpenTableForRead(os.path.join(vis, 'DATA_DESCRIPTION')) as tb:
            self.dd_spw_map = tb.getcol('SPECTRAL_WINDOW_ID')
            self.dd_pol_map = tb.getcol('POLARIZATION_ID')

        # read spw information (channel freq, channel width, freq_ref)
        with casa.OpenTableForRead(os.path.join(vis, 'SPECTRAL_WINDOW')) as tb:
            meas_freq_ref = tb.getcol('MEAS_FREQ_REF')
            self.chan_freq = {}
            self.chan_width = {}
            self.freq_ref = {}
            for irow in range(tb.nrows()):
                self.freq_ref[irow] = self.freq_ref_string(meas_freq_ref[irow])
                self.chan_freq[irow] = tb.getcell('CHAN_FREQ', irow)
                self.chan_width[irow] = tb.getcell('CHAN_WIDTH', irow)

        # read field information
        with casa.OpenTableForRead(os.path.join(vis, 'FIELD')) as tb:
            self.field_dir = {}
            measinfo = tb.getcolkeyword('PHASE_DIR', 'MEASINFO')
            if 'Ref' in measinfo:
                self.field_ref = measinfo['Ref']
            else:
                # variable reference column, assume J2000
                self.field_ref = 'J2000'
            for irow in range(tb.nrows()):
                self.field_dir[irow] = tb.getcell('PHASE_DIR', irow)

            all_fields = np.arange(tb.nrows(), dtype=np.int64)

        # nominal image LSRK frequency (will be used for channel selection)
        # calculate based on
        #     - ALMA observatory position
        #     - nominal field direction (one of the TARGET)
        #     - observation start time (from OBSERVATION table?)
        # observation start time
        with casa.OpenTableForRead(os.path.join(vis, 'OBSERVATION')) as tb:
            time_range = tb.getcell('TIME_RANGE', 0)
            obs_start_time = time_range[0]
        # field id
        with casa.OpenMSMetaData(vis) as msmd:
            try:
                #field_ids = msmd.fieldsforintent(intent='OBSERVE_TARGET#ON_SOURCE*')
                field_ids = msmd.fieldsforintent(intent=self.visparam.intent)
                if len(field_ids) == 0:
                    field_ids = all_fields
            except Exception:
                field_ids = all_fields
            #nominal_field_id = field_ids[0]
        # data description id
        data_desc_ids = np.arange(len(self.dd_spw_map))
        _times = np.empty(len(data_desc_ids), dtype=np.float64)
        _times[:] = obs_start_time
        _field_ids = np.empty_like(data_desc_ids)
        self.nominal_lsr_frequency = {}
        for field_id in field_ids:
            _field_ids[:] = field_id
            cf = self._get_lsr_frequency(_times,
                                         data_desc_ids,
                                         _field_ids)
            self.nominal_lsr_frequency[field_id] = cf

    def get_lsr_frequency(self, chunk):
        # sanity check
        # - all chunk entry should have same timestamp (mitigate in future?)
        assert np.all(chunk['time'] == chunk['time'][0])
        # - all chunk entry should have same spw (mitigate in future?)
        assert np.all(chunk['data_desc_id'] == chunk['data_desc_id'][0])
        # - all chunk entry should have same field (mitigate in fugure?)
        assert np.all(chunk['field_id'] == chunk['field_id'][0])

        # TODO: rewrite _get_lsr_frequency with the assumption that
        #       time and data_desc_id is constant over the chunk
        times = chunk['time'][:1]
        data_desc_ids = chunk['data_desc_id'][:1]
        field_ids = chunk['field_id'][:1]

        cf_lsr = self._get_lsr_frequency(times, data_desc_ids, field_ids)
        return cf_lsr[0]

    def _get_lsr_frequency(self, times, data_desc_ids, field_ids):
        # sanity check
        # - consistency of chunk data length
        assert len(times) == len(data_desc_ids)
        assert len(times) == len(field_ids)

        me = casa.CreateCasaMeasure()
        qa = casa.CreateCasaQuantity()
        # position measure -- observatory position of ALMA
        me.doframe(me.observatory('ALMA'))

        nchunk = len(times)

        lsr_frequency = {}
        lsr_width = {}
        for i in range(nchunk):
            ddid = data_desc_ids[i]
            field_id = field_ids[i]
            spwid = self.dd_spw_map[ddid]
            freq_ref = self.freq_ref[spwid]
            chan_freq = self.chan_freq[spwid]
            chan_width = self.chan_width[spwid]
            nchan = len(chan_freq)
            lsr_freq_edge = np.empty(nchan + 1, dtype=np.float64)
            lsr_freq_edge[:nchan] = chan_freq - chan_width / 2.0
            lsr_freq_edge[nchan] = chan_freq[-1] + chan_width[-1] / 2.0
            if freq_ref == 'LSRK':
                # frequency is in LSRK, no conversion is needed
                pass
            else:
                # require conversion to LSRK
                # time measure
                epoch = qa.quantity(times[i], 's')
                mepoch = me.epoch('utc', epoch)
                me.doframe(mepoch)
                # direction measure
                field_dir = self.field_dir[field_id][:,0]
                lon = qa.quantity(field_dir[0], 'rad')
                lat = qa.quantity(field_dir[1], 'rad')
                mdirection = me.direction(self.field_ref, lon, lat)
                me.doframe(mdirection)
                # frequency measure
                frequency = qa.quantity(0.0, 'Hz')
                mfrequency = me.frequency(freq_ref, frequency)
                for ichan in range(nchan):
                    mfrequency['m0']['value'] = lsr_freq_edge[ichan]#chan_freq[ichan] - 0.5 * chan_width[ichan]
                    #print 'LOG mfrequency = {0}'.format(mfrequency)
                    # convert
                    converted = me.measure(mfrequency, 'LSRK')
                    lsr_freq_edge[ichan] = converted['m0']['value']
                mfrequency['m0']['value'] = lsr_freq_edge[nchan]#chan_freq[nchan-1] + 0.5 * chan_width[nchan-1]
                converted = me.measure(mfrequency, 'LSRK')
                lsr_freq_edge[nchan] = converted['m0']['value']
            lsr_frequency[i] = lsr_freq_edge
                #print 'LOG    native = {0}'.format(chan_freq)
                #print 'LOG converted = {0}'.format(lsr_frequency[i])
        #return lsr_frequency, lsr_width
        return lsr_frequency

    def fill_data(self, ws, chunk, lsr_edge_frequency, datacolumn):
        qa = casa.CreateCasaQuantity()

        lsr_frequency = (lsr_edge_frequency[1:] + lsr_edge_frequency[:-1]) / 2.0

        # info from chunk
        field_id = chunk['field_id'][0]
        data_desc_id = chunk['data_desc_id'][0]

        # get spectral channel selection parameter
        start = self.imageparam.start
        width = self.imageparam.width
        nchan = self.imageparam.nchan
        npol = chunk[datacolumn].shape[0]
        nrow = chunk[datacolumn].shape[2]
        qstart = qa.quantity(start)
        qwidth = qa.quantity(width)
        qnchan = qa.quantity(nchan)
        start_unit = qa.getunit(qstart)
        width_unit = qa.getunit(qwidth)
        frequency_pattern = re.compile('^(G|k|M|T)?Hz$')
        wavelength_pattern = re.compile('^(k|c|m|n)?m$')
        velocity_pattern = re.compile(wavelength_pattern.pattern.replace('$', '/s$'))
        match_with = lambda pattern: pattern.match(start_unit) is not None and \
                                        pattern.match(width_unit) is not None

        image_freq = np.empty(nchan, dtype=np.float64)
        image_width = 0.0

        # get image LSRK frequency
        # start, width, and nchan are defined as follows:
        #
        #   start=<center frequency of first image channel>
        #     |
        # |-------|-------|-------| nchan=3
        # |<----->|
        #   width=<constant channel width of image channel>
        #
        if len(start_unit) == 0 and len(width_unit) == 0:
            # channel selection
            # use nominal LSRK frequency for image (channel boundary)
            nominal_lsr_frequency = self.nominal_lsr_frequency[field_id][data_desc_id]
            #nominal_lsr_width = self.nominal_lsr_width[field_id][data_desc_id]
            if nchan == 1 and width == -1:
                # this is special setting that maps all visibility channels into
                # single image channel
                image_freq[0] = (nominal_lsr_frequency[-1] + nominal_lsr_frequency[0]) / 2.0
                image_width = nominal_lsr_frequency[-1] - nominal_lsr_frequency[0]
            else:
                for ichan in range(nchan):
                    # left boundary of start channel
                    channel_start = int(qstart['value']) + ichan * int(qwidth['value'])
                    # right boundary of end channel
                    channel_end = channel_start + int(qwidth['value'])
                    # center frequency of the range
                    image_freq[ichan] = (nominal_lsr_frequency[channel_start] +
                                         nominal_lsr_frequency[channel_end]) / 2.0
                image_width = (nominal_lsr_frequency[1] - nominal_lsr_frequency[0]) \
                                * int(qwidth['value'])
        elif match_with(frequency_pattern):
            # frequency selection
            for ichan in range(nchan):
                image_freq[ichan] = qa.convert(
                    qa.add(qstart, qa.mul(qwidth, ichan)),
                    'Hz')['value']
            image_width = qa.convert(qwidth, 'Hz')['value']
        elif match_with(wavelength_pattern):
            # wavelength selection -- not supported yet
            raise NotImplementedError('wavelength selection is not implemented yet')
        elif match_with(velocity_pattern):
            # velocity selection -- not supported yet
            raise NotImplementedError('velocity selection is not implemented yet')
        else:
            # invalid or mixed selection
            raise ValueError('image channel selection is invalid or not supported')

        # map/interpolate
        data_desc_ids = chunk['data_desc_id']
        nchunk = len(data_desc_ids)
        spwid = self.dd_spw_map[data_desc_id]
        chan_width = self.chan_width[spwid][0]

        # chunk holds WEIGHT column, which should store channelized weight,
        # 2 * df * dt (which equals 2 * EFFECTIVE_BW[0] * INTEVAL) so that
        # weight scaling factor is 1.0 by default
        weight_factor = 1.0
        #print 'LOG: SPW {0} chan_width * 2 = {1}, image_width = {2}'.format(
        #    spwid, chan_width * 2, image_width)
        if abs(image_width) < 1.99 * abs(chan_width):
            #print 'LOG: do interpolation'
            # interpolation
            # TODO: array shape (order) should be compatible with sakura gridding function
            ws_shape = (nrow, 1, nchan,)
            ws_shape2 = (nrow, nchan,)
            real = sakura.empty_aligned(ws_shape, dtype=np.float32)
            imag = sakura.empty_aligned(ws_shape, dtype=np.float32)
            flag = sakura.empty_aligned(ws_shape, dtype=bool)
            weight = sakura.empty_aligned(ws_shape2, dtype=np.float32)
            row_flag = sakura.empty_aligned((nrow,), dtype=bool)
            channel_map = sakura.empty_aligned((nchan,), dtype=np.int32)
            channel_map[:] = np.arange(nchan, dtype=np.int32)

            # real, image: linear interpolation
            #print 'LOG: lsr_frequency length {0} real.shape {1}'.format(
            #    len(lsr_frequency), chunk[datacolumn].shape)
            if chunk[datacolumn].shape[1] > 1:
                data_interp = interpolate.interp1d(lsr_frequency, chunk[datacolumn],
                                                   kind='linear', axis=1,
                                                   fill_value='extrapolate')
                _data = data_interp(image_freq)
                # flag: nearest interpolation
                flag_interp = interpolate.interp1d(lsr_frequency, chunk['flag'],
                                                   kind='nearest', axis=1,
                                                   fill_value='extrapolate')
                _flag = flag_interp(image_freq)
            else:
                _data = chunk[datacolumn]
                _flag = chunk['flag']

            _weight = chunk['weight']
            self._to_stokesI(_data, _flag, _weight, weight_factor, real, imag, flag, weight)
        else:
            #print 'LOG: do channel mapping'
            # channel mapping
            # if chunk frequency for i goes into image frequency cell for k,
            # i maps into k
            image_freq_boundary = np.empty(nchan + 1, dtype=np.float64)
            image_freq_boundary[:-1] = image_freq - 0.5 * image_width
            image_freq_boundary[-1] = image_freq[-1] + 0.5 * image_width
            image_freq_min = image_freq_boundary.min()
            image_freq_max = image_freq_boundary.max()
            in_range = np.where(
                np.logical_and(
                    image_freq_min <= lsr_frequency,
                    lsr_frequency <= image_freq_max))[0]
            nvischan = len(in_range)
            # accumulate N channels improves weight factor by N
            weight_factor *= float(nvischan)
            # TODO: array shape (order) should be compatible with sakura gridding function
            ws_shape = (nrow, 1, nvischan,)
            ws_shape2 = (nrow, nvischan,)
            real = sakura.empty_aligned(ws_shape, dtype=np.float32)
            imag = sakura.empty_aligned(ws_shape, dtype=np.float32)
            flag = sakura.empty_aligned(ws_shape, dtype=bool)
            weight = sakura.empty_aligned(ws_shape2, dtype=np.float32)
            row_flag = sakura.empty_aligned((nrow,), dtype=bool)
            channel_map = sakura.empty_aligned((nvischan,), dtype=np.int32)
            for ichan, chan_chunk in enumerate(in_range):
                # fill in channel_map
                f = lsr_frequency[chan_chunk]
                for chan_image in range(nchan):
                    b0 = image_freq_boundary[chan_image]
                    b1 = image_freq_boundary[chan_image + 1]
                    if b0 > b1:
                        tmp = b1
                        b1 = b0
                        b0 = tmp
                    if b0 <= f and f <= b1:
                        channel_map[ichan] = chan_image
                        break

                # fill in data
                _data = chunk[datacolumn][:, chan_chunk:chan_chunk + 1, :]
                _flag = chunk['flag'][:, chan_chunk:chan_chunk + 1, :]
                _weight = chunk['weight']
                self._to_stokesI(_data, _flag, _weight, weight_factor,
                                 real[:, :, ichan:ichan + 1], imag[:, :, ichan:ichan + 1],
                                 flag[:, :, ichan:ichan + 1], weight[:, ichan:ichan + 1])

        # row_flag
        row_flag[:] = np.all(flag == True, axis=(1, 2,))

        # invert flag
        # reader definition:
        #     True: INVALID
        #     False: *VALID*
        # working set definition:
        #     True: *VALID*
        #     False: INVALID
        np.logical_not(row_flag, row_flag)
        np.logical_not(flag, flag)

        ws.rdata = real
        ws.idata = imag
        ws.flag = flag
        ws.row_flag = row_flag
        ws.weight = weight
        ws.channel_map = channel_map

    def _check_phasecenter(self, phasecenter):
        if isinstance(phasecenter, int):
            # integer that should indicate field ID
            return
        elif isinstance(phasecenter, str):
            # string
            if phasecenter.isdigit():
                # all characters are digits, it should indicate field ID
                return
            else:
                # string representation of
                raise ValueError('Invalid phasecenter value: \"{0}\".'.format(phasecenter)
                                 + 'Currently arbitrary phasecenter is not supported.'
                                 + 'Please specify field ID instead.')
        else:
            raise ValueError('Invalid phasecenter value: \"{0}\".'.format(phasecenter)
                             + 'Currently arbitrary phasecenter is not supported.'
                             + 'Please specify field ID instead.')

    def _warn_refocus(self):
        print('***WARN*** refocusing is disabled even if distance to the source is known.')

    def fill_uvw(self, ws, chunk, lsr_edge_frequency):
        """
        Fill UV coordinate

        ws -- working set to be filled
        chunk -- input data chunk
        lsr_edge_frequency -- channel edge frequency (LSRK)
        """
        phasecenter = self.imageparam.phasecenter
        self._check_phasecenter(phasecenter)
        #self._warn_refocus()

        qa = casa.CreateCasaQuantity()
        speed_of_light = qa.constants('c')
        c = qa.convert(speed_of_light, 'm/s')['value']
        data_shape = chunk['flag'].shape
        nrow = data_shape[2]
        nchan = data_shape[1]
        uvw = chunk['uvw']
        data_desc_ids = chunk['data_desc_id']
        uvgrid = self.imageparam.uvgridconfig
        delta_u = uvgrid.cellu
        delta_v = uvgrid.cellv
        offset_u = uvgrid.offsetu
        offset_v = uvgrid.offsetv

        # UVW conversion
        u = sakura.empty_aligned((nrow, nchan), dtype=uvw.dtype)
        v = sakura.empty_like_aligned(u)
        center_freq = np.asfarray([np.mean(lsr_edge_frequency[i:i+2]) for i in range(nchan)])
        for irow in range(nrow):
            # TODO: phase rotation if image phasecenter is different from
            #       the reference direction of the observation
            pass

            # conversion from physical baseline length to the value
            # normalized by observing wavelength
            # u0 = uvw[0, irow]
            # v0 = uvw[1, irow]
            spw_id = data_desc_ids[irow]
            #chan_freq = lsr_frequency
            #chan_width = (lsr_edge_frequency[1:] - lsr_edge_frequency[:-1]).mean()
            #freq_start = chan_freq[0] - chan_width / 2
            #freq_end = chan_freq[-1] + chan_width / 2
            #freq_start = lsr_edge_frequency[0]
            #freq_end = lsr_edge_frequency[-1]
            #center_freq = (freq_start + freq_end) / 2
            #center_freq = np.frombuffer(np.mean([(lsr_edge_frequency[i:i+2]) for i in range(nchan)]), dtype=np.float64)

            # u[irow] = u0 * center_freq / c  # divided by wavelength
            # v[irow] = v0 * center_freq / c  # divided by wavelength

            # TODO?: refocus UVW if distance to the source is known
            pass

            # project uv-coordinate value onto gridding pixel plane
            # pixel size is determined by an inverse of image extent
            # along x (RA) and y (DEC) direction
            #
            #    v
            #     nv-1|(nmin,vmax)      (umax,vmax)
            #        .|
            #        .|
            #        .|
            #        .|
            #     nv/2|           (0,0)
            #        .|
            #        .|
            #        2|
            #        1|
            #        0|(umin,vmin)      (umax,vmin)
            #         |__________________________
            #          0 1 2 ......nu/2...nu-1 u
            # u[irow] = u[irow] / delta_u + offset_u
            # u[irow] = (u0 * center_freq) / (delta_u * c) + offset_u
            u[irow] = ( uvw[0, irow] * center_freq) / (delta_u * c) + offset_u

            # Sign of v must be inverted so that MFISTA routine generates
            # proper image. Otherwise, image will be flipped in the vertical
            # axis.
            # v[irow] = -v[irow] / delta_v + offset_v
            # v[irow] = -(v0 * center_freq) / (delta_v * c) + offset_v
            v[irow] = -( uvw[1, irow] * center_freq) / (delta_v * c) + offset_v

        ws.u = u
        ws.v = v


    def flatten(self, working_set):
        """
        Generator yielding list of working_sets divided by spectral channels
        """
        nchan = working_set.nchan
        nrow_ws = working_set.nrow
        chan_image = np.unique(working_set.channel_map)
        num_ws = len(chan_image)
        for imchan in chan_image:
            vischans = np.where(working_set.channel_map == imchan)[0]
            nchan = len(vischans)
            nrow = nrow_ws * nchan
            ws_shape = (nrow, 1, 1,)
            ws_shape2 = (nrow, 1,)
            real = sakura.empty_aligned(ws_shape, dtype=working_set.rdata.dtype)
            imag = sakura.empty_aligned(ws_shape, dtype=working_set.idata.dtype)
            flag = sakura.empty_aligned(ws_shape, dtype=working_set.flag.dtype)
            weight = sakura.empty_aligned(ws_shape2, dtype=working_set.weight.dtype)
            row_flag = sakura.empty_aligned((nrow,), dtype=working_set.row_flag.dtype)
            channel_map = sakura.empty_aligned((1,), dtype=working_set.channel_map.dtype)
            u = sakura.empty_aligned((nrow,), dtype=working_set.u.dtype)
            v = sakura.empty_like_aligned(u)
            row_start = 0
            for ichan in vischans:
                row_end = row_start + nrow_ws
                real[row_start:row_end, 0, 0] = working_set.rdata[:, 0, ichan]
                imag[row_start:row_end, 0, 0] = working_set.idata[:, 0, ichan]
                flag[row_start:row_end, 0, 0] = working_set.flag[:, 0, ichan]
                weight[row_start:row_end, 0] = working_set.weight[:, ichan]
                row_flag[row_start:row_end] = working_set.row_flag[:]
                channel_map[0] = imchan
                u[row_start:row_end] = working_set.u[:, ichan]
                v[row_start:row_end] = working_set.v[:, ichan]
                row_start = row_end

            ws = gridder.GridderWorkingSet(data_id=working_set.data_id, u=u, v=v,
                                           rdata=real, idata=imag, flag=flag,
                                           weight=weight, row_flag=row_flag,
                                           channel_map=channel_map)
            #print('yielding channelized working set from channels {}'.format(vischans))
            yield ws

    def generate_working_set(self, chunk):
        """
        generate working set for gridder from the given data chunk
        that is supposed to be produced by VisibilityReader.

        Procedure to generate working set is as follows:
            1. visibility frequency conversion (from TOPO to LSRK in ALMA)
            2. channel mapping between raw visibility in working set to
               visibility grid coordinate or visibility interpolation
            3. rotate phase if image phasecenter is different from
               reference direction of observation
            4. refocus uvw if distance to source is known (maybe optional)
            5. convert UVW in metre into uv(w) coordinate

        chunk -- data chunk produced by readvis
        """
        # sanity check
        # - chunk should contain all required data
        candidate_datacolumns = set(('data', 'corrected_data', 'residual_data'))
        for column in self.required_columns:
            if column == 'data':
                assert any(col in chunk for col in candidate_datacolumns)
            else:
                assert column in chunk
        # - all chunk entry should have same timestamp (mitigate in future?)
        assert np.all(chunk['time'] == chunk['time'][0])
        # - all chunk entry should have same spw (mitigate in future?)
        assert np.all(chunk['data_desc_id'] == chunk['data_desc_id'][0])

        #print 'Chunk ID {0} is valid'.format(chunk['chunk_id'])

        # working set to be filled in
        chunk_id = chunk['chunk_id']
        working_set = gridder.GridderWorkingSet(data_id=chunk_id)
        #print('LOG: generate working set for visibility chunk #{0}'.format(chunk_id))

        # 1. visibility frequency conversion
        # get LSRK frequency at channel boundary
        # one LSRK frequency for one chunk
        lsr_frequency = self.get_lsr_frequency(chunk)
        # returned frequency is for channel boundary so its length
        # should be greater than 1
        assert len(lsr_frequency) > 1

        # 2. channel mapping or regridding (i.e., fill data/flag/weight
        #    with channel map. interpolating data if necessary)
        available_datacolumns = set(chunk.keys()).intersection(candidate_datacolumns)
        assert len(available_datacolumns) == 1
        datacolumn = available_datacolumns.pop()
        self.fill_data(working_set, chunk, lsr_frequency, datacolumn)

        # 3~5. UVW manipulation
        self.fill_uvw(working_set, chunk, lsr_frequency)

        # EXTRA. convert channelized working set into a set of
        #        single-channel working set
        working_set_list = list(self.flatten(working_set))

        #print 'Working set data shape {0} polmap {1}'.format(working_set.rdata.shape,
        #                                                     working_set.pol_map)

        return working_set_list

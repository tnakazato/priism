from __future__ import absolute_import

import numpy
import os
import shutil
import time

import priism.alma.paramcontainer as paramcontainer
import priism.alma.visreader as visreader
import priism.alma.visconverter as visconverter
import priism.external.casa as casa
import priism.external.sakura as sakura
from . import utils


class VisibilityConverterTest(utils.TestBase):
    """
    Test suite for visconverter

        test_arbitrary_phasecenter: test arbitrary phasecenter specification
                                    (exception case)
        test_channel_interp: test image frequency axis specification by channel
                             (interpolation case)
        test_channel_map:    test image frequency axis specification by channel
                             (mapping case)
        test_freq_interp:    test image frequency axis specification by frequency
                             (interpolation case)
        test_freq_map:       test image frequency axis specification by frequency
                             (mapping case)
        test_parallel: test parallel map
        test_lsr_freq_topo: test frequency conversion from TOPO to LSRK
                            (returned frequency is channel boundary)
        test_lsr_freq_lsrk: test skipping frequency conversion if LSRK
                            (returned frequency is channel boundary)
    """
    imagename = 'VisibilityConverterTest.fits'
    result = None
    datapath = os.path.join(os.path.expanduser('~'),
                            'casadevel/data/regression/unittest/mstransform')
    vis = 'test-subtables-alma.ms'

    def setUp(self):
        print('setUp: copying {0}...'.format(self.vis))
        shutil.copytree(os.path.join(self.datapath, self.vis), self.vis)

        self.assertTrue(os.path.exists(self.vis))

        with casa.OpenTableForRead(os.path.join(self.vis, 'DATA_DESCRIPTION')) as tb:
            self.dd_spw_map = tb.getcol('SPECTRAL_WINDOW_ID')
            self.dd_pol_map = tb.getcol('POLARIZATION_ID')

        with casa.OpenTableForRead(os.path.join(self.vis, 'SPECTRAL_WINDOW')) as tb:
            self.nchans = tb.getcol('NUM_CHAN')

        with casa.OpenTableForRead(os.path.join(self.vis, 'POLARIZATION')) as tb:
            self.npols = tb.getcol('NUM_CORR')

        # avaliable FIELD_ID in MAIN table is only 6
        self.field_id = 6

        # default visparam
        self.visparam = paramcontainer.VisParamContainer(vis=self.vis)

    def tearDown(self):
        if os.path.exists(self.vis):
            print('tearDown: removing {0}...'.format(self.vis))
            shutil.rmtree(self.vis)

        self.assertFalse(os.path.exists(self.vis))

    def test_arbitrary_phasecenter(self):
        # create data chunk
        nrow = 1
        dd_id = 0
        spw_id = self.dd_spw_map[dd_id]
        pol_id = self.dd_pol_map[dd_id]
        nchan = self.nchans[spw_id]
        npol = self.npols[pol_id]
        chunk = self.get_chunk_template(nrow=nrow, dd_id=dd_id)

        # create converter
        image_nchan = 1
        image_start = 0
        image_width = 1
        converter = self.get_converter(start=image_start,
                                       width=image_width,
                                       nchan=image_nchan)

        # set phasecenter to unsupported value
        converter.imageparam.phasecenter = ''

        with self.assertRaises(ValueError) as cm:
            ws = converter.generate_working_set(chunk)
        the_exception = cm.exception
        self.assertTrue(str(the_exception).startswith('Invalid phasecenter value'))

    def test_channel_map(self):
        # create data chunk
        nrow = 1
        dd_id = 0
        spw_id = self.dd_spw_map[dd_id]
        pol_id = self.dd_pol_map[dd_id]
        nchan = self.nchans[spw_id]
        npol = self.npols[pol_id]
        chunk = self.get_chunk_template(nrow=nrow, dd_id=dd_id)

        # create converter
        image_nchan = 1
        image_start = 0
        image_width = 10
        converter = self.get_converter(start=image_start,
                                       width=image_width,
                                       nchan=image_nchan)

        ws = converter.generate_working_set(chunk)

        rdata = ws.rdata
        data_shape = rdata.shape

        # consistency check
        self.verify_ws_consistency(ws, chunk['chunk_id'])

        expected_shape = (nrow, 1, image_nchan * image_width)
        self.assertEqual(data_shape, expected_shape)

        start = image_start
        end = image_start + image_nchan * image_width
        expected_data = numpy.zeros(expected_shape, dtype=numpy.complex)
        expected_flag = numpy.ones(expected_shape, dtype=numpy.bool)
        for ipol in range(npol):
            expected_data += (chunk['data'][ipol:ipol + 1, start:end, :] \
                * numpy.where(chunk['flag'][ipol:ipol + 1, start:end, :] is False, 0.5, 0.0)).transpose((2, 0, 1))
            expected_flag = numpy.logical_and(expected_flag, chunk['flag'][ipol:ipol + 1, start:end, :])
        expected_flag = numpy.logical_not(expected_flag)
        eps = 1e-7
        self.assertMaxDiffLess(expected_data.real, ws.rdata, eps)
        self.assertMaxDiffLess(expected_data.imag, ws.idata, eps)
        self.assertTrue(numpy.all(expected_flag == ws.flag))

        for ichan in range(image_nchan):
            s = ichan * image_width
            e = s + image_width
            self.assertTrue(numpy.all(ws.channel_map[s:e] == ichan))

    def test_channel_interp(self):
        # create data chunk
        nrow = 1
        dd_id = 0
        spw_id = self.dd_spw_map[dd_id]
        pol_id = self.dd_pol_map[dd_id]
        nchan = self.nchans[spw_id]
        npol = self.npols[pol_id]
        chunk = self.get_chunk_template(nrow=nrow, dd_id=dd_id)

        # here UVW is set to 0
        # this should not happen in interferometry observation
        # but just for testing purpose...
        chunk['uvw'][:] = 0.0

        # timestamp should be start time of the observation
        # for LSRK conversion
        with casa.OpenTableForRead(os.path.join(self.vis, 'OBSERVATION')) as tb:
            time_range = tb.getcell('TIME_RANGE', 0)
        chunk['time'][:] = time_range[0]

        # create converter
        image_nchan = 1
        image_start = 0
        image_width = 1
        converter = self.get_converter(start=image_start,
                                       width=image_width,
                                       nchan=image_nchan)

        ws = converter.generate_working_set(chunk)

        rdata = ws.rdata
        data_shape = rdata.shape

        # consistency check
        self.verify_ws_consistency(ws, chunk['chunk_id'])

        expected_shape = (1, image_nchan * image_width, nrow)
        self.assertEqual(data_shape, expected_shape)

        d0 = chunk['data'][:, 0, 0]
        f0 = chunk['flag'][:, 0, 0]
        base_data = d0
        base_flag = f0
        expected_data = numpy.complex(0)
        expected_flag = True
        eps = 1e-5
        for ipol in range(npol):
            if base_flag[ipol] == False:
                expected_data += 0.5 * base_data[ipol]
            expected_flag = numpy.logical_and(expected_flag, base_flag[ipol])
        self.assertMaxDiffLess(expected_data.real, ws.rdata[0, 0, 0], eps)
        self.assertMaxDiffLess(expected_data.imag, ws.idata[0, 0, 0], eps)
        ref = numpy.logical_not(expected_flag)
        val = ws.flag[0, 0, 0]
        self.assertEqual(ref, val)
        self.assertEqual(ws.channel_map[0], 0)

        # (u,v) = (0,0) maps onto (nearly) grid plane center
        self.assertTrue(ws.u[0] == int(self.imageparam.imsize[0]) // 2)
        self.assertTrue(ws.v[0] == int(self.imageparam.imsize[1]) // 2)

    def test_freq_interp(self):
        # create data chunk
        nrow = 1
        dd_id = 0
        spw_id = self.dd_spw_map[dd_id]
        pol_id = self.dd_pol_map[dd_id]
        nchan = self.nchans[spw_id]
        npol = self.npols[pol_id]
        chunk = self.get_chunk_template(nrow=nrow, dd_id=dd_id)

        # create converter
        timestamp = chunk['time'][0]
        field_id = self.field_id
        mfr = 'TOPO'
        spw_id = self.dd_spw_map[dd_id]
        with casa.OpenTableForRead(os.path.join(self.vis, 'SPECTRAL_WINDOW')) as tb:
            chan_freq = tb.getcell('CHAN_FREQ', spw_id)
        lsr_freq = self._convert(timestamp, field_id, mfr, chan_freq)
        # start channel is lsr frequency that corresponds to 1/3 of first channel
        # channel width is native one
        image_nchan = 1
        f = lsr_freq[0] + (lsr_freq[1] - lsr_freq[0]) / 3.0
        image_start = '{0}Hz'.format(f)
        image_width = '{0}Hz'.format(lsr_freq[1] - lsr_freq[0])
        converter = self.get_converter(start=image_start,
                                       width=image_width,
                                       nchan=image_nchan)

        # uvw from expected pixel value
        expected_u = 70
        expected_v = 25
        self._uv_from_pixel(expected_u, expected_v,
                            chunk['uvw'][:, 0], lsr_freq)

        # conversion
        ws = converter.generate_working_set(chunk)

        rdata = ws.rdata
        data_shape = rdata.shape

        # consistency check
        self.verify_ws_consistency(ws, chunk['chunk_id'])

        expected_shape = (1, image_nchan, nrow)
        self.assertEqual(data_shape, expected_shape)

        print('LOG: rdata={0}'.format(ws.rdata))
        print('LOG: idata={0}'.format(ws.idata))
        print('LOG: chunk data0 {0} 1 {1}'.format(chunk['data'][:, 0, 0], chunk['data'][:, 1, 0]))

        d0 = chunk['data'][:, 0, 0]
        d1 = chunk['data'][:, 1, 0]
        base_data = (2.0 * d0 + 1.0 * d1) / 3.0
        f0 = chunk['flag'][:, 0, 0]
        f1 = chunk['flag'][:, 0, 0]
        base_flag = f0
        expected_data = numpy.zeros(expected_shape, dtype=numpy.complex)
        expected_flag = numpy.ones(expected_shape, dtype=numpy.bool)
        eps = 1e-5
        for ipol in range(npol):
            if base_flag[ipol] == False:
                expected_data += 0.5 * base_data[ipol]
            expected_flag = numpy.logical_and(expected_flag, base_flag[ipol])
        self.assertMaxDiffLess(expected_data.real, ws.rdata[0, 0, 0], eps)
        self.assertMaxDiffLess(expected_data.imag, ws.idata[0, 0, 0], eps)
        ref = numpy.logical_not(expected_flag)
        val = ws.flag[0, 0, 0]
        self.assertEqual(ref, val)
        self.assertEqual(ws.channel_map[0], 0)

        # TODO: uv test
        print(expected_u, ws.u[0])
        print(expected_v, ws.v[0])
        eps = 1.0e-5
        self.assertMaxDiffLess(expected_u, ws.u[0], eps)
        self.assertMaxDiffLess(expected_v, ws.v[0], eps)

    def test_freq_map(self):
        # create data chunk
        nrow = 1
        dd_id = 0
        spw_id = self.dd_spw_map[dd_id]
        pol_id = self.dd_pol_map[dd_id]
        nchan = self.nchans[spw_id]
        npol = self.npols[pol_id]
        chunk = self.get_chunk_template(nrow=nrow, dd_id=dd_id)

        # create converter
        timestamp = chunk['time'][0]
        field_id = self.field_id
        mfr = 'TOPO'
        spw_id = self.dd_spw_map[dd_id]
        with casa.OpenTableForRead(os.path.join(self.vis, 'SPECTRAL_WINDOW')) as tb:
            chan_freq = tb.getcell('CHAN_FREQ', spw_id)
        lsr_freq = self._convert(timestamp, field_id, mfr, chan_freq)
        # start channel and width are chosen so that first 10 visibility
        # channels are included
        image_nchan = 1
        f = (lsr_freq[4] + lsr_freq[5]) / 2.0
        image_start = '{0}Hz'.format(f)
        image_width = '{0}Hz'.format(lsr_freq[10] - lsr_freq[0])
        converter = self.get_converter(start=image_start,
                                       width=image_width,
                                       nchan=image_nchan)

        # uvw from expected pixel value
        expected_u = 0
        expected_v = 0
        self._uv_from_pixel(expected_u, expected_v,
                            chunk['uvw'][:, 0], lsr_freq)

        # conversion
        ws = converter.generate_working_set(chunk)

        rdata = ws.rdata
        data_shape = rdata.shape

        # consistency check
        self.verify_ws_consistency(ws, chunk['chunk_id'])

        expected_shape = (nrow, 1, 10)
        self.assertEqual(data_shape, expected_shape)

        start = 0
        end = start + image_nchan * 10
        expected_data = numpy.zeros(expected_shape, dtype=numpy.complex)
        expected_flag = numpy.ones(expected_shape, dtype=numpy.bool)
        for ipol in range(npol):
            expected_data += (chunk['data'][ipol:ipol + 1, start:end, :] \
                * numpy.where(chunk['flag'][ipol:ipol + 1, start:end, :] is False, 0.5, 0.0)).transpose((2, 0, 1))
            expected_flag = numpy.logical_and(expected_flag, chunk['flag'][ipol:ipol + 1, start:end, :])
        eps = 1e-7
        self.assertMaxDiffLess(expected_data.real, ws.rdata, eps)
        self.assertMaxDiffLess(expected_data.imag, ws.idata, eps)
        self.assertTrue(numpy.all(numpy.logical_not(expected_flag) == ws.flag))
        for ichan in range(image_nchan):
            s = ichan * 10
            e = s + 10
            self.assertTrue(numpy.all(ws.channel_map[s:e] == ichan))

        # uv test
        # u,v should ideally be zero but allow small value within
        # the one indicated by tolerance, eps
        eps = 1.0e-5
        self.assertLess(abs(ws.u[0]), eps)
        self.assertLess(abs(ws.v[0]), eps)

    def test_parallel(self):
        # create reader
        reader = visreader.VisibilityReader(self.visparam)

        # create converter
        start = 3
        width = 10
        nchan = 1
        converter = self.get_converter(start, width, nchan)

        max_nrow = 16

        num_measure = 10
        serial_results = numpy.empty(num_measure, dtype=numpy.float32)
        parallel_results = numpy.empty_like(serial_results)
        conv = lambda chunk: (chunk['chunk_id'], converter.generate_working_set(chunk))
        for i in range(num_measure):
            # serial run
            ws_serial = []
            start_serial = time.time()
            for chunk_id, ws in map(conv,
                                    reader.readvis(nrow=max_nrow)):
                ws_serial.append((chunk_id, ws))
            end_serial = time.time()

            serial_results[i] = end_serial - start_serial
            print('SERIAL RUN: {0} sec'.format(serial_results[i]))

            # parallel run
            num_threads = 2
            ws_parallel = []
            start_parallel = time.time()
            for chunk_id, ws in sakura.paraMap(num_threads,
                                               conv,
                                               reader.readvis(nrow=max_nrow)):
                ws_parallel.append((chunk_id, ws))
            end_parallel = time.time()

            parallel_results[i] = end_parallel - start_parallel
            print('PARALLEL RUN: {0} sec'.format(parallel_results[i]))

            # consistency check
            for chunk_id, ws in ws_serial:
                self.verify_ws_consistency(ws, chunk_id)
            for chunk_id, ws in ws_parallel:
                self.verify_ws_consistency(ws, chunk_id)
            self.assertEqual(len(ws_serial), len(ws_parallel))
            for _, p in ws_parallel:
                found = False
                for i in range(len(ws_serial)):
                    _, s = ws_serial[i]
                    if p.data_id == s.data_id:
                        _, s = ws_serial.pop(i)
                        self.assertTrue(numpy.all(p.u == s.u))
                        self.assertTrue(numpy.all(p.v == s.v))
                        self.assertTrue(numpy.all(p.rdata == s.rdata))
                        self.assertTrue(numpy.all(p.idata == s.idata))
                        self.assertTrue(numpy.all(p.flag == s.flag))
                        self.assertTrue(numpy.all(p.row_flag == s.row_flag))
                        self.assertTrue(numpy.all(p.weight == s.weight))
                        self.assertTrue(numpy.all(p.channel_map == s.channel_map))
                        found = True
                        break
                self.assertTrue(found)


        # verification
        print('LOG: SERIAL RUN: {0}'.format(serial_results))
        print('LOG: PARALLEL RUN: {0}'.format(parallel_results))
        acceleration = serial_results / parallel_results
        print('LOG: ACCELERATION: {0} (max {1} min {2})'.format(acceleration,
                                                                acceleration.max(),
                                                                acceleration.min()))
        self.assertLess(parallel_results.min(), serial_results.min())

    def get_chunk_template(self, nrow=1, dd_id=0):
        chunk = {}
        spw_id = self.dd_spw_map[dd_id]
        pol_id = self.dd_pol_map[dd_id]
        nchan = self.nchans[spw_id]
        npol = self.npols[pol_id]
        chunk['chunk_id'] = 0
        chunk['data_desc_id'] = numpy.empty(nrow, dtype=numpy.int32)
        chunk['field_id'] = numpy.empty_like(chunk['data_desc_id'])
        chunk['data_desc_id'][:] = dd_id
        chunk['field_id'][:] = self.field_id
        with casa.SelectTableForRead(self.vis,
                'DATA_DESC_ID=={0} && ANTENNA1 != ANTENNA2'.format(dd_id)) as tb:
            chunk['time'] = tb.getcol('TIME', 0, nrow)
            chunk['data'] = tb.getcol('DATA', 0, nrow)
            chunk['flag'] = tb.getcol('FLAG', 0, nrow)
            chunk['uvw'] = tb.getcol('UVW', 0, nrow)
            chunk['weight'] = tb.getcol('WEIGHT', 0, nrow)

        return chunk

    def get_converter(self, start, width, nchan):
        self.imageparam = paramcontainer.ImageParamContainer(imagename=self.imagename,
                                                             start=start,
                                                             width=width,
                                                             nchan=nchan,
                                                             phasecenter=str(self.field_id))
        print('LOG: s,w,n {0} {1} {2}'.format(self.imageparam.start,
                                              self.imageparam.width,
                                              self.imageparam.nchan))
        converter = visconverter.VisibilityConverter(self.visparam, self.imageparam)
        return converter

#     def assertTrue(self, expr, msg=None):
#         try:
#             super(VisibilityConverterTest, self).assertTrue(expr, msg)
#         except Exception, e:
#             import pydevd
#             pydevd.settrace()
#             raise e
#
    def verify_ws_consistency(self, ws, chunk_id):
        # consistency check
        for attr in ('data_id', 'rdata', 'idata', 'flag', 'weight', 'u', 'v', 'channel_map', 'row_flag'):
            self.assertTrue(hasattr(ws, attr), msg='attr {0} is missing'.format(attr))
            val = getattr(ws, attr)
            if attr == 'data_id':
                self.assertEqual(val, chunk_id)
                continue
            self.assertTrue(hasattr(val, 'base'))
            base = getattr(val, 'base')
            base_str = base.__str__()
            self.assertTrue(base_str.find('libsakurapy.AlignedPyArray') != -1)
        data_shape = ws.rdata.shape
        self.assertEqual(ws.idata.shape, data_shape)
        self.assertEqual(ws.flag.shape, data_shape)
        self.assertEqual(len(ws.u), data_shape[0])
        self.assertEqual(len(ws.v), data_shape[0])
        self.assertEqual(ws.weight.shape, (data_shape[0], data_shape[2],))
        self.assertEqual(len(ws.channel_map), data_shape[2])
        self.assertEqual(len(ws.row_flag), data_shape[0])

        print('LOG: ws data_shape {0}'.format(data_shape))
        print('LOG: u = {0}'.format(ws.u.tolist()))
        print('LOG: v = {0}'.format(ws.v.tolist()))

    def test_lsr_freq_topo(self):
        self._test_lsr_freq(mfr='TOPO')

    def test_lsr_freq_lsrk(self):
        self._test_lsr_freq(mfr='LSRK')

    def _test_lsr_freq(self, mfr='TOPO'):
        chunk = {}
        ndds = len(self.dd_spw_map)
        with casa.OpenTableForRead(os.path.join(self.vis, 'FIELD')) as tb:
            nfields = tb.nrows()
        with casa.OpenTableForReadWrite(os.path.join(self.vis, 'SPECTRAL_WINDOW')) as tb:
            mfrs = tb.getcol('MEAS_FREQ_REF')
            self.assertTrue(mfr in visconverter.VisibilityConverter.frequency_reference)
            mfr_id = visconverter.VisibilityConverter.frequency_reference.index(mfr)
            if not numpy.all(mfrs == mfr_id):
                mfrs[:] = mfr_id
                tb.putcol('MEAS_FREQ_REF', mfrs)

        visparam = paramcontainer.VisParamContainer(vis=self.vis)
        imageparam = paramcontainer.ImageParamContainer(imagename=self.imagename,
                                                        phasecenter=str(self.field_id))
        converter = visconverter.VisibilityConverter(visparam, imageparam)

        nchunk = 1
        chunk['time'] = numpy.empty(nchunk, dtype=numpy.float64)
        chunk['data_desc_id'] = numpy.empty(nchunk, dtype=numpy.int32)
        chunk['field_id'] = numpy.empty_like(chunk['data_desc_id'])
        with casa.OpenTableForRead(self.vis) as tb:
            chunk['time'][:] = tb.getcell('TIME', 0)
        for field_id in range(nfields):
            chunk['field_id'][0] = field_id
            for dd_id in range(ndds):
                chunk['data_desc_id'][0] = dd_id
                lsr_freq = converter.get_lsr_frequency(chunk)
                #print lsr_freq
                spw_id = self.dd_spw_map[dd_id]
                nchan = self.nchans[spw_id]
                self.assertEqual(nchan + 1, len(lsr_freq))
                self.assertEqual(mfrs[spw_id], mfr_id)

                # special case for LSRK
                with casa.OpenTableForRead(os.path.join(self.vis, 'SPECTRAL_WINDOW')) as tb:
                    chan_freq = tb.getcell('CHAN_FREQ', spw_id)
                    chan_width = tb.getcell('CHAN_WIDTH', spw_id)
                expected = numpy.empty(nchan + 1, dtype=numpy.float64)
                expected[:-1] = chan_freq - chan_width / 2.0
                expected[-1] = chan_freq[-1] + chan_width[-1] / 2.0
                if mfr == 'LSRK':
                    self.assertTrue(numpy.all(lsr_freq == expected))
                else:
                    expected = self._convert(chunk['time'][0],
                                             field_id, mfr, expected)
                    self.assertMaxDiffLess(expected, lsr_freq, 1e-15)

    def _convert(self, timestamp, field_id, mfr, chan_freq):
        me = casa.CreateCasaMeasure()
        qa = casa.CreateCasaQuantity()
        vis = self.vis
        with casa.OpenTableForRead(os.path.join(vis, 'FIELD')) as tb:
            reference_dir = tb.getcell('PHASE_DIR', field_id)[:, 0]
            reference_frame = 'J2000' # hard coded
        lon = qa.quantity(reference_dir[0], 'rad')
        lat = qa.quantity(reference_dir[1], 'rad')
        mdirection = me.direction(rf=reference_frame, v0=lon, v1=lat)
        mepoch = me.epoch(rf='UTC', v0=qa.quantity(timestamp, 's'))
        mposition = me.observatory('ALMA')
        me.doframe(mdirection)
        me.doframe(mepoch)
        me.doframe(mposition)
        lsr_freq = numpy.empty_like(chan_freq)
        for ichan in range(len(lsr_freq)):
            mfrequency = me.frequency(rf=mfr, v0=qa.quantity(chan_freq[ichan], 'Hz'))
            converted = me.measure(mfrequency, rf='LSRK')
            lsr_freq[ichan] = converted['m0']['value']
        return lsr_freq

    def _uv_from_pixel(self, px, py, uvw, lsr_freq):
        qa = casa.CreateCasaQuantity()
        nx = self.imageparam.imsize[0]
        ny = self.imageparam.imsize[1]
        dx = qa.convert(self.imageparam.cell[0], 'rad')['value']
        dy = qa.convert(self.imageparam.cell[1], 'rad')['value']
        wx = nx * dx
        wy = ny * dy
        offx = int(nx) // 2
        offy = int(ny) // 2
        fcenter = (lsr_freq.min() + lsr_freq.max()) / 2.0
        c = qa.convert(qa.constants('c'), 'm/s')['value']
        lcenter = c / fcenter
        uvw[0] = lcenter * (px - offx) / wx
        uvw[1] = lcenter * (py - offy) / wy


def suite():
    test_items = ['test_channel_interp',
                  'test_channel_map',
                  'test_freq_interp',
                  'test_freq_map',
                  'test_lsr_freq_topo',
                  'test_lsr_freq_lsrk',
                  'test_arbitrary_phasecenter',
                  'test_parallel']
    test_suite = utils.generate_suite(VisibilityConverterTest,
                                      test_items)
    return test_suite

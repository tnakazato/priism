from __future__ import absolute_import

import unittest
import numpy
import itertools

import almasparsemodeling.core as core
import almasparsemodeling.external.sakura as sakura
import almasparsemodeling.test.utils as utils

class GridderTest(utils.TestBase):
    """
    Test suite for gridder
    
    test_position_center -- grid position test (center)
    test_position_bottom_left -- grid position test (bottom left)
    test_position_bottom_right -- grid position test (bottom right)
    test_position_top_left -- grid position test (top left)
    test_position_top_right -- grid position test (top right)
    test_position_polarization -- grid position test (polarization axis) SKIP
    test_position_channel -- grid position test (spectral axis)
    test_gridfunction_box -- test BOX gridfunction
    test_gridfunction_sf -- test GAUSSIAN gridfunction
    test_gridfunction_gauss -- test SF (prolate-Spheroidal) gridfunction
    test_polarization_map -- test polarization mapping 
    test_channel_map -- test channel mapping
    test_flag -- test channelized flag handling
    test_row_flag -- test row flag handling
    test_weight -- test weight handling
    test_weight_pol -- test weight handling along polarization axis
    test_multi_ws -- test gridding multiple ws
    """
    def setUp(self):
        super(GridderTest, self).setUp()
        
        self.standard_imparam = core.ImageParamContainer(nchan=1,
                                                         imsize=[5,7])
        convsupport = 1
        convsampling = 10
        gridfunction = core.GridFunctionUtil.box(convsupport, convsampling)
        print 'support {0} sampling {1} len(gridfunction) {2}'.format(convsupport,
                                                                      convsampling,
                                                                      len(gridfunction))
        self.standard_gridparam = core.GridParamContainer(convsupport=convsupport,
                                                          convsampling=convsampling,
                                                          gridfunction=gridfunction)
    
    def tearDown(self):
        super(GridderTest, self).tearDown()
        
        del self.standard_imparam
        del self.standard_gridparam
        
    def _generate_ws_template(self, nrow=1, nchan=1, npol=1):
        u = sakura.empty_aligned((nrow,), dtype=numpy.float64)
        v = sakura.empty_like_aligned(u)
        rdata = sakura.empty_aligned((nrow, npol, nchan,), dtype=numpy.float32)
        idata = sakura.empty_like_aligned(rdata)
        flag = sakura.empty_aligned(rdata.shape, dtype=numpy.bool)
        weight = sakura.empty_aligned((nrow, nchan,), dtype=numpy.float32)
        row_flag = sakura.empty_aligned(u.shape, dtype=numpy.bool)
        channel_map = sakura.empty_aligned((nchan,), dtype=numpy.int32)
        ws = core.GridderWorkingSet(data_id=0,
                                    u=u,
                                    v=v,
                                    rdata=rdata,
                                    idata=idata,
                                    flag=flag,
                                    weight=weight,
                                    row_flag=row_flag,
                                    channel_map=channel_map) 
        
        # set defualt value for some attributes
        ws.flag[:] = True
        ws.row_flag[:] = True
        ws.weight[:] = 1.0
        ws.channel_map[:] = range(nchan)
        
        return ws
    
    def _configure_uv(self, gridder, ws, ws_row=0, loc='center'):
        """
        in (nu, nv) pixels, pixel layout is as follows:
        
            (top left)               (top right)
            |(nv-1,0)|(nv-1,1)|...|(nv-1,nu-1)|
            ...
            |(1,0)|(1,1)|(1,2)|...|(1,nu-1)|
            |(0,0)|(0,1)|(0,2)|...|(0,nu-1)|
            (bottom left)            (bottom right)
            
        memory layout is as follows:
        
             (top left)               (top right)
            |(nv-1)*nu|(nv-1)*nu+1|...|nv*nu-1|
            ...
            |nu|nu+1|nu+2|............|2*nu-1|
            |0|1|2|...................|nu-1|
            (bottom left)            (bottom right)
           
        """
        imageparam = gridder.imageparam
        u = -1.0
        v = -1.0
        if loc == 'center':
            u = float(imageparam.imsize[0] - 1) / 2
            v = float(imageparam.imsize[1] - 1) / 2
        elif loc == 'top_left':
            u = 0.0
            v = float(imageparam.imsize[1] - 1)
        elif loc == 'bottom_left':
            u = 0.0
            v = 0.0
        elif loc == 'top_right':
            u = float(imageparam.imsize[0] - 1)
            v = float(imageparam.imsize[1] - 1)
        elif loc == 'bottom_right':
            u = float(imageparam.imsize[0] - 1)
            v = 0.0
        else:
            self.fail('invalid location mode: {}'.format(loc))
        ws.u[ws_row] = u
        ws.v[ws_row] = v
        
    def _configure_data(self, ws, rdata, idata, weight=None, flag=None, row_flag=None):
        nelem = ws.nrow * ws.npol * ws.nchan
        assert len(rdata) >= nelem
        assert len(idata) >= nelem
        if flag is not None:
            assert len(flag) >= nelem
        if weight is not None:
            assert len(weight) >= ws.nrow * ws.nchan
        if row_flag is not None:
            assert len(row_flag) >= ws.nrow
        
        for irow in xrange(ws.nrow):
            for ipol in xrange(ws.npol):
                start = irow * ws.npol * ws.nchan + ipol * ws.nchan
                end = start + ws.nchan
                ws.rdata[irow,ipol,:] = rdata[start:end]
                ws.idata[irow,ipol,:] = idata[start:end]
                if flag is not None:
                    ws.flag[irow,ipol,:] = flag[start:end]
            if weight is not None:
                start = irow * ws.nchan
                end = start + ws.nchan
                ws.weight[irow,:] = weight[start:end]
        
        if row_flag is not None:
            ws.row_flag[:] = row_flag
            
    
    def _generate_gridder(self, imparam=None, gridparam=None):
        if imparam is None:
            myimparam = self.standard_imparam
        else:
            myimparam = imparam
        if gridparam is None:
            mygridparam = self.standard_gridparam
        else:
            mygridparam = gridparam
            
        gridder = core.VisibilityGridder(mygridparam, myimparam)
        return gridder
    
    def _verify_shape(self, gridder, result):
        expected_shape = (gridder.nv, gridder.nu, gridder.npol, gridder.nchan)
        self.assertEqual(result.real.shape, expected_shape)
        self.assertEqual(result.imag.shape, expected_shape)
    
    def _verify_nonzero(self, data, index_list):
        #nonzero_pixels = numpy.where(data != 0.0)
        eps = 1.0e-7
        nonzero_pixels = numpy.where(abs(data) >= eps)
        ndx = numpy.lexsort(numpy.asarray(nonzero_pixels))
        nonzero_pixels = (nonzero_pixels[0][ndx],
                          nonzero_pixels[1][ndx],
                          nonzero_pixels[2][ndx],
                          nonzero_pixels[3][ndx],)
        print 'nonzero: {}'.format(nonzero_pixels)
        print 'index_list: {}'.format(index_list)
        self.assertIndexListEqual(nonzero_pixels, index_list)
        return nonzero_pixels
    
#     def _verify_data(self, result_real, result_imag, ref_real, ref_imag):
#         self.assertEqual(len(result_real), len(ref_real))
#         self.assertEqual(len(result_imag), len(ref_imag))
#         ereal = ref_real
#         eimag = ref_imag
#         print 'EXPECTED', ereal, eimag
#         print 'ACTUAL', result_real, result_imag
#         eps = 1.0e-7
#         self.assertMaxDiffLess(ereal, result_real, eps)
#         self.assertMaxDiffLess(eimag, result_imag, eps)
        
    def _verify_data(self, result, ref):
        print 'EXPECTED', ref
        print 'ACTUAL', result

        self.assertEqual(len(result), len(ref))
        if len(result) == 0:
            return
        
        eps = 1.0e-7
        sorted_result = numpy.sort(result)
        sorted_ref = numpy.sort(ref)
        self.assertMaxDiffLess(sorted_ref, sorted_result, eps)
        
    def _grid_and_verify(self, gridder, ws, loc_real, loc_imag, ref_real, ref_imag): 
        if hasattr(ws, '__iter__'):
            ws_list = ws
        else:
            ws_list = [ws]
            
        for _ws in ws_list:
            # grid data
            for irow in xrange(_ws.nrow):
                print 'rdata {0}\tidata {1}\tu {2}\tv {3}'.format(_ws.rdata[irow].flatten(), 
                                                                  _ws.idata[irow].flatten(),
                                                                  _ws.u[irow],
                                                                  _ws.v[irow])
            #print 'gridfunction={}'.format(gridder.gridfunction)
            gridder.grid(_ws)
        
        # verification
        result = gridder.get_result()
        print 'result_imag:', result.imag
    
        # number of ws's accumulated onto grid
        self.assertEqual(result.num_ws, len(ws_list))
        
        # check grid shape
        self._verify_shape(gridder, result)
        
        # check nonzero pixel
        ndx_real = numpy.lexsort(numpy.asarray(loc_real))
        #print 'ndx_real', ndx_real
        loc_real = (numpy.asarray(loc_real[0])[ndx_real],
                    numpy.asarray(loc_real[1])[ndx_real],
                    numpy.asarray(loc_real[2])[ndx_real],
                    numpy.asarray(loc_real[3])[ndx_real],)
        ndx_imag = numpy.lexsort(numpy.asarray(loc_imag))
        #print 'ndx_imag', ndx_imag
        if len(ndx_imag) > 0:
            loc_imag = (numpy.asarray(loc_imag[0])[ndx_imag],
                        numpy.asarray(loc_imag[1])[ndx_imag],
                        numpy.asarray(loc_imag[2])[ndx_imag],
                        numpy.asarray(loc_imag[3])[ndx_imag],)
        nonzero_pixels_real = self._verify_nonzero(result.real, loc_real)
        nonzero_pixels_imag = self._verify_nonzero(result.imag, loc_imag)
        #self.assertIndexListEqual(nonzero_pixels_real, nonzero_pixels_imag)
        #self.assertTrue(len(nonzero_pixels_real[0]) == len(expected_location[0]))
        
        greal = result.real[nonzero_pixels_real]
        gimag = result.imag[nonzero_pixels_imag]
        ref_real = ref_real[ndx_real]
        if len(ndx_imag) > 0:
            ref_imag = ref_imag[ndx_imag]
        print 'greal:', greal
        print 'ref_real:', ref_real
        print 'gimag:', gimag
        print 'ref_imag:', ref_imag

        self._verify_data(greal, ref_real)
        self._verify_data(gimag, ref_imag)
           
    
    def _test_position_one(self, locobj):
        """
        test template for simple gridding test
        
            locobj -- dictionary whose key sets 'location mode' while 
                      corresponding value is expected location which is 
                      four tuple describing ([pu], [pv], [ppol], [pchan]).
                      available modes are:
                          center
                          bottom_left

        """
        # parse locobj
        self.assertEqual(len(locobj), 1)
        location_mode = locobj.iterkeys().next()
        # sort axes: (u, v, pol, chan) -> (v, u, pol, chan)
        a, b, c, d = locobj[location_mode]
        #expected_location = (b, a, c, d)
        
        # gridder
        gridder = self._generate_gridder()
        
        # working set
        ws = self._generate_ws_template(1, 1, 1)
        self._configure_data(ws, [1.0], [0.1])
        self._configure_uv(gridder, ws, 0, loc=location_mode)
        rdata = numpy.asarray([ws.rdata[0,0,0]])
        idata = numpy.asarray([ws.idata[0,0,0]])
        weight = numpy.asarray([ws.weight[0,0]])
        
        # perform gridding and verification
        ref_real = rdata * weight / weight
        ref_imag = idata * weight / weight
        
        if location_mode == 'center':
            loc_real = (b, a, c, d,)#expected_location
            loc_imag = ([], [], [], [])
            ref_imag = []
        else:
            cen = (float(gridder.nu - 1) / 2, float(gridder.nv - 1) / 2)
            loc_real = ([b[0], cen[1] + (cen[1] - b[0])],
                        [a[0], cen[0] + (cen[0] - a[0])],
                        [c[0], c[0]], [d[0], d[0]],)
            loc_imag = loc_real
            ref_real = numpy.asarray([ref_real[0], ref_real[0]])
            ref_imag = numpy.asarray([ref_imag[0], -ref_imag[0]])
        
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)

    def test_position_center(self):
        """test_position_center: grid position test (center)"""
        locobj = {'center': ([2], [3], [0], [0])}
        self._test_position_one(locobj)
        
    def test_position_bottom_left(self):
        """test_position_bottom_left: grid position test (bottom left)"""
        locobj = {'bottom_left': ([0], [0], [0], [0])}
        self._test_position_one(locobj)
       
    def test_position_top_left(self):
        """test_position_top_left: grid position test (top left)"""
        locobj = {'top_left': ([0], [6], [0], [0])}
        self._test_position_one(locobj)
       
    def test_position_bottom_right(self):
        """test_position_bottom_right: grid position test (bottom right)"""
        locobj = {'bottom_right': ([4], [0], [0], [0])}
        self._test_position_one(locobj)
       
    def test_position_top_right(self):
        """test_position_top_right: grid position test (top right)"""
        locobj = {'top_right': ([4], [6], [0], [0])}
        self._test_position_one(locobj)
        
    def test_position_polarization(self):
        """test_position_polarization: grid position test (polarization axis)"""
        self.skipTest('grid position test for polarization axis is skipped ' 
                      'since polarization axis is constrained to 1.')
        
    def test_position_channel(self):
        """test_position_channel: grid position test (spectral axis)"""
        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata, idata, weight)
        
        # perform gridding and verification
        loc_real = ([3,3,6,6,0,0], [2,2,4,4,0,0], [0,0,0,0,0,0], [0,1,0,1,0,1])
        loc_imag = ([6,6,0,0], [4,4,0,0], [0,0,0,0], [0,1,0,1])
        ref_real = rdata * weight / weight
        ref_imag = (idata * weight / weight)[2:]
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)

    def test_gridder_polarization_map(self):
        """test_gridder_polarization_map: test polarization mapping"""
        nchan = 1
        nrow = 2
        npol = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=npol, nchan=nchan)
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 1.0, 1.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata[:4], idata[:4], weight[:2])

        # customize polarization map
        ws.pol_map[:] = 0
        
        # perform gridding and verification
        loc_real = ([3,6,0], [2,4,0], [0,0,0], [0,0,0])
        loc_imag = ([6,0], [4,0], [0,0], [0,0])
        ref_real = numpy.zeros(len(loc_real[0]), dtype=numpy.float32)
        ref_imag = numpy.zeros_like(ref_real)
        ones = numpy.ones(npol, dtype=numpy.float32)
        ref_real[0] = numpy.sum(ws.rdata[0] * ws.weight[0,0]) / numpy.sum(ones * ws.weight[0,0])
        ref_real[1] = numpy.sum(ws.rdata[1] * ws.weight[1,0]) / numpy.sum(ones * ws.weight[1,0])
        ref_real[2] = ref_real[1]
        ref_imag[0] = numpy.sum(ws.idata[0] * ws.weight[0,0]) / numpy.sum(ones * ws.weight[0,0])
        ref_imag[1] = numpy.sum(ws.idata[1] * ws.weight[1,0]) / numpy.sum(ones * ws.weight[1,0])
        ref_imag[2] = -ref_imag[1]
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag[1:])

    def test_gridder_channel_map(self):
        """test_gridder_channel_map: test channel mapping"""
        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=1,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        
        # customize channel map
        ws.channel_map[:] = 0
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata[:4], idata[:4], weight[:4])
        
        # perform gridding and verification
        loc_real = ([3,6,0], [2,4,0], [0,0,0], [0,0,0])
        loc_imag = ([6,0], [4,0], [0,0], [0,0])
        ref_real = numpy.zeros(len(loc_real[0]), dtype=numpy.float32)
        ref_imag = numpy.zeros_like(ref_real)
        ref_real[0] = numpy.sum(ws.rdata[0] * ws.weight[0]) / numpy.sum(ws.weight[0])
        ref_real[1] = numpy.sum(ws.rdata[1] * ws.weight[1]) / numpy.sum(ws.weight[1])
        ref_real[2] = ref_real[1]
        ref_imag[0] = numpy.sum(ws.idata[0] * ws.weight[0]) / numpy.sum(ws.weight[0])
        ref_imag[1] = numpy.sum(ws.idata[1] * ws.weight[1]) / numpy.sum(ws.weight[0])
        ref_imag[2] = -ref_imag[1]
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag[1:])

    def test_flag(self):
        """test_flag: test channelized flag handling"""
        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata[:4], idata[:4], weight[:4], 
                             flag=[True, False, True, True])

        # perform gridding and verification
        loc_real = ([3,6,6,0,0], [2,4,4,0,0], [0,0,0,0,0], [0,0,1,0,1])
        loc_imag = ([6,6,0,0], [4,4,0,0], [0,0,0,0], [0,1,0,1])
        ref_real = (rdata * weight / weight).take([0,2,3,4,5])
        ref_imag = (idata * weight / weight).take([2,3,4,5])
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)

    def test_row_flag(self):
        """test_row_flag: test row flag handling"""
        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata[:4], idata[:4], weight[:4], 
                             row_flag=[True, False])
        
        # perform gridding and verification
        loc_real = ([3,3], [2,2], [0,0], [0,1])
        loc_imag = ([], [], [], [])
        ref_real = (rdata * weight / weight)[:2]
        ref_imag = []
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)
        
    def test_weight(self):
        """test_weight: test weight handling"""
        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata[:4], idata[:4], weight[:4])
        
        # perform gridding and verification
        loc_real = ([3,6,0], [2,4,0], [0,0,0], [0,1,1])
        loc_imag = ([6,0], [4,0], [0,0], [1,1])
        nz = weight.nonzero()
        ref_real = rdata[nz] * weight[nz] / weight[nz]
        ref_imag = (idata[nz] * weight[nz] / weight[nz])[1:]
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)

    def test_weight_pol(self):
        """test_weight_pol: test weight handling along polarization axis"""
        nchan = 1
        nrow = 2
        npol = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=npol, nchan=nchan)
        
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        weight = numpy.asarray([1.0, 0.0, 0.0])
        
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        self._configure_uv(gridder, ws, 1, loc='top_right')
        self._configure_data(ws, rdata[:4], idata[:4], weight[:2])
        
        # customize polarization map
        ws.pol_map[:] = 0
        
        # perform gridding and verification
        loc_real = ([3], [2], [0], [0])
        loc_imag = ([], [], [], [])
        ref_real = numpy.zeros(len(loc_real[0]), dtype=numpy.float32)
        ref_imag = []
        ones = numpy.ones(npol, dtype=numpy.float32)
        ref_real[0] = numpy.sum(ws.rdata[0] * ws.weight[0,0]) / numpy.sum(ones * ws.weight[0,0])
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)

    def test_multi_ws(self):
        """test_multi_ws: test gridding multiple ws"""
        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        # gridder
        gridder = self._generate_gridder(imparam=imparam)
        
        # working set
        nws = 2
        ws1 = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        ws2 = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
        
        # expected data
        rdata1 = numpy.asarray([1.0, 2.0, 10.0, 20.0, 10.0, 20.0])
        rdata2 = rdata1 * 0.25
        idata1 = numpy.asarray([0.1, 0.2, 0.01, 0.02, -0.01, -0.02])
        idata2 = idata1 * 5.0
        weight1 = numpy.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weight2 = weight1 * 0.5
        
        # configure ws1
        self._configure_uv(gridder, ws1, 0, loc='center')
        self._configure_uv(gridder, ws1, 1, loc='top_right')
        self._configure_data(ws1, rdata1[:4], idata1[:4], weight1[:4])
       
        # configure ws2
        self._configure_uv(gridder, ws2, 0, loc='center')
        self._configure_uv(gridder, ws2, 1, loc='top_right')
        self._configure_data(ws2, rdata2[:4], idata2[:4], weight2[:4])

        # perform gridding and verification
        loc_real = ([3,3,6,6,0,0], [2,2,4,4,0,0], [0,0,0,0,0,0], [0,1,0,1,0,1])
        loc_imag = ([6,6,0,0], [4,4,0,0], [0,0,0,0], [0,1,0,1])
        ref_real = ((rdata1 * weight1 + rdata2 * weight2) / (weight1 + weight2))
        ref_imag = ((idata1 * weight1 + idata2 * weight2) / (weight1 + weight2))[2:]
        self._grid_and_verify(gridder, [ws1, ws2], loc_real, loc_imag, 
                              ref_real, ref_imag)
        
    def _test_gridfunction(self, gtype, support, sampling=100, gparam=None):
        params = {'convsupport': support,
                  'convsampling': sampling}
        if isinstance(gparam, dict):
            params.update(gparam)
        gridfunction = getattr(core.GridFunctionUtil, gtype)(**params)
        print 'Custom Grid Fucntion:'
        print 'support {0} sampling {1} len(gridfunction) {2}'.format(support,
                                                                      sampling,
                                                                      len(gridfunction))
        gridparam = core.GridParamContainer(convsupport=support, convsampling=sampling,
                                            gridfunction=gridfunction)

        nchan = 2
        nrow = 2
        imparam = core.ImageParamContainer(nchan=nchan,
                                           imsize=self.standard_imparam.imsize)
        
        # gridder
        gridder = self._generate_gridder(imparam=imparam, gridparam=gridparam)
        
        # working set
        ws = self._generate_ws_template(nrow=nrow, npol=1, nchan=nchan)
         
        # expected data
        rdata = numpy.asarray([1.0, 2.0, 10.0, 20.0])
        idata = numpy.asarray([0.1, 0.2, 0.01, 0.02])
        weight = numpy.asarray([1.0, 1.0, 1.0, 1.0])
         
        # configure ws
        self._configure_uv(gridder, ws, 0, loc='center')
        # shift uv: distance from pixel center is 0.707 (=sqrt(2) * 0.5)
        ws.u[0] += 0.51
        ws.v[0] += 0.51
        self._configure_uv(gridder, ws, 1, loc='center')
        # shift uv: distance from pixel center is 0.5
        ws.u[1] += 0.51
        self._configure_data(ws, rdata, idata, weight)
         
        # perform gridding and verification
        w = len(gridfunction.nonzero()[0]) / sampling
        iw = int(w)
        print w
        #umin = int(round(ws.u[0] - iw))
        #umax = int(ws.u[0] + iw)
        cen = (float(gridder.nu - 1) / 2, float(gridder.nv - 1) / 2)
        print 'cen', cen, 'nunv', gridder.nu, gridder.nv
        print 'u:', ws.u
        print 'v:', ws.v
        umin = 0
        umax = gridder.nu
        vmin = 0
        vmax = gridder.nv
        lv = range(vmin, vmax) * gridder.nu
        lv.sort()
        lv = lv + lv
        lu = range(umin, umax)
        lu = lu * gridder.nv
        lu = lu + lu
        lp = [0] * len(lv)
        lc = [0] * (len(lv) / 2) + [1] * (len(lv) / 2)
        #print 'lu, lv, lp, lc=', lu, lv, lp, lc
        loc_real = (lv, lu, lp, lc)
        ref_real = numpy.zeros(len(loc_real[0]), dtype=numpy.float32)
        ref_imag = numpy.zeros_like(ref_real)
        w_real = numpy.zeros_like(ref_real)
        w_imag = numpy.zeros_like(ref_real)
        
        def _get_kernel_index(du, dv, sampling):
            dist = numpy.sqrt(du * du + dv * dv) * numpy.float64(sampling)
            idist = int(dist)
            if dist - idist >= 1.0 - 1e-10:
                print 'Increment idist since dist-int(dist) > 1'
                idist += 1
            return idist
        
        def _accumulate(greal, gimag, wreal, wimag, gindex, 
                        dreal, dimag, dweight, dindex,
                        gridfunction, idist, conj=False):
            gr = 0.0
            gi = 0.0
            wr = 0.0
            wi = 0.0
            if idist < len(gridfunction) - 1:
                sign = -1 if conj else 1
                k = gridfunction[idist]
                print 'accumulate', k, dreal[dindex], dimag[dindex], dweight[dindex]
                gr += dreal[dindex] * k * dweight[dindex]
                gi += sign * dimag[dindex] * k * dweight[dindex]
                wr += k * dweight[dindex]
                wi += k * dweight[dindex]
            return gr, gi, wr, wi
        
        for i in xrange(len(ref_real)):
            x = lu[i]
            y = lv[i]
            p = lp[i]
            c = lc[i]
            du1 = ws.u[0] - x
            dv1 = ws.v[0] - y
            du2 = ws.u[1] - x
            dv2 = ws.v[1] - y
            idist1 = _get_kernel_index(du1, dv1, sampling)
            idist2 = _get_kernel_index(du2, dv2, sampling)

            self.assertGreaterEqual(idist1, 0)
            self.assertGreaterEqual(idist2, 0)
            #print 'dist', dist1, dist2
            gr1, gi1, wr1, wi1 = _accumulate(ref_real, ref_imag, w_real, w_imag, i, 
                                         rdata, idata, weight, c, 
                                         gridfunction, idist1, False)
            gr2, gi2, wr2, wi2 = _accumulate(ref_real, ref_imag, w_real, w_imag, i, 
                                         rdata, idata, weight, 2+c, 
                                         gridfunction, idist2, False)
            ref_real[i] += gr1 + gr2
            ref_imag[i] += gi1 + gi2
            w_real[i] += wr1 + wr2
            w_imag[i] += wi1 + wi2            

            # grid complex conjugate
            du1 = cen[0] + (cen[0] - ws.u[0]) - x
            dv1 = cen[1] + (cen[1] - ws.v[0]) - y
            du2 = cen[0] + (cen[0] - ws.u[1]) - x
            dv2 = cen[1] + (cen[1] - ws.v[1]) - y
            idist1 = _get_kernel_index(du1, dv1, sampling)
            idist2 = _get_kernel_index(du2, dv2, sampling)
            self.assertGreaterEqual(idist1, 0)
            self.assertGreaterEqual(idist2, 0)
            gr1, gi1, wr1, wi1 = _accumulate(ref_real, ref_imag, w_real, w_imag, i, 
                                         rdata, idata, weight, c, 
                                         gridfunction, idist1, True)
            gr2, gi2, wr2, wi2 = _accumulate(ref_real, ref_imag, w_real, w_imag, i, 
                                         rdata, idata, weight, 2+c, 
                                         gridfunction, idist2, True)
            ref_real[i] += gr1 + gr2
            ref_imag[i] += gi1 + gi2
            w_real[i] += wr1 + wr2
            w_imag[i] += wi1 + wi2
                
            if w_real[i] > 0.0:
                ref_real[i] /= w_real[i]
            if w_imag[i] > 0.0:
                ref_imag[i] /= w_imag[i]
        
        eps = 1.0e-7
        ndx_real = []
        for i in xrange(len(ref_real)):
            if abs(ref_real[i]) != 0.0:
                ndx_real.append(i)
        print 'ndx_real', ndx_real
        loc_real = (numpy.asarray(lv)[ndx_real], numpy.asarray(lu)[ndx_real],
                    numpy.asarray(lp)[ndx_real], numpy.asarray(lc)[ndx_real])
        ref_real = ref_real[ndx_real]
        ndx_imag = []
        for i in xrange(len(ref_imag)):
            if abs(ref_imag[i]) >= eps:
                ndx_imag.append(i)
        print 'ndx_imag', ndx_imag
        loc_imag = (numpy.asarray(lv)[ndx_imag], numpy.asarray(lu)[ndx_imag],
                    numpy.asarray(lp)[ndx_imag], numpy.asarray(lc)[ndx_imag])
        ref_imag = ref_imag[ndx_imag]
        self._grid_and_verify(gridder, ws, loc_real, loc_imag, ref_real, ref_imag)
       
    def test_gridfunction_box(self):
        """test_gridfunction_box: test BOX gridfunction"""
        self._test_gridfunction('box', 1)
    
    def test_gridfunction_gauss(self):
        """test_gridfunction_gauss: test GAUSSIAN gridfunction"""
        self._test_gridfunction('gauss', 1, gparam={'hwhm': 0.5})
    
    def test_gridfunction_sf(self):
        """test_gridfunction_sf: test SF (prolate-Spheroidal) gridfunction"""
        self._test_gridfunction('sf', 1)

def suite():
    test_items = ['test_position_center',
                  'test_position_bottom_left',
                  'test_position_top_left',
                  'test_position_bottom_right',
                  'test_position_top_right',
                  'test_position_polarization',
                  'test_gridder_polarization_map',
                  'test_gridder_channel_map',
                  'test_position_channel',
                  'test_flag',
                  'test_row_flag',
                  'test_weight',
                  'test_weight_pol',
                  'test_multi_ws',
                  'test_gridfunction_box',
                  'test_gridfunction_gauss',
                  'test_gridfunction_sf']
    test_suite = utils.generate_suite(GridderTest,
                                      test_items)
    return test_suite
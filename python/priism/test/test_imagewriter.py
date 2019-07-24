from __future__ import absolute_import

import numpy
import os

import priism.alma.paramcontainer as paramcontainer
import priism.alma.imagewriter as imagewriter
import priism.external.casa as casa
import priism.test.utils as utils


class ImageWriterTest(utils.TestBase):
    """
    Test suite for imagewriter

       test_default_meta: imagewriter test with default image metadata
       test_coordsys: imagewriter coordsys handling test
       test_phasecenter: imagewriter phasecenter handling test
    """

    imagename = 'ImageWriterTest.fits'
    result = None

    def setUp(self):
        pass

    def tearDown(self):
        #if os.path.exists(self.imagename):
        #    os.remove(self.imagename)

        #self.assertFalse(os.path.exists(self.imagename))
        pass

    def _get_reference(self, csys, reference_name, type, asdict=False):
        refdict = getattr(csys, reference_name)(type=type)
        #print '{0} for {1}: {2}'.format(reference_name, type, refdict)
        if asdict:
            return refdict
        else:
            return refdict['numeric']

    def assertClose(self, actual, expected, eps=1.0e-11):
        if expected == 0.0:
            diff = abs(actual)
        else:
            diff = abs((actual - expected) / expected)
        self.assertLess(diff, eps,
                        msg='actual {0} expected {1} (diff {2}'.format(actual,
                                                                       expected,
                                                                       diff))

    def _run_test(self, imageparam, arr, imagemeta=None):
        qa = casa.CreateCasaQuantity()

        writer = imagewriter.ImageWriter(imageparam,
                                         arr,
                                         imagemeta)

        if imagemeta is None:
            # default image metadata
            imagemeta = paramcontainer.ImageMetaInfoContainer()

        status = writer.write(overwrite=True)

        self.assertTrue(status)
        self.assertTrue(os.path.exists(self.imagename))

        self.result = self.imagename

        ia = casa.CreateCasaImageAnalysis()
        ia.open(self.imagename)
        try:
            # image shape
            expected_shape = (imageparam.imsize[0],
                              imageparam.imsize[1],
                              imageparam.nchan,
                              1)
            self.assertTrue(numpy.all(expected_shape == ia.shape()))
            csys = ia.coordsys()

            # direction coordinate
            dirref = csys.referencecode(type='direction')[0]
            self.assertEqual(dirref, 'J2000')
            refpix = self._get_reference(csys, 'referencepixel', 'direction')
            refval = self._get_reference(csys, 'referencevalue', 'direction')
            incr = self._get_reference(csys, 'increment', 'direction')

            self.assertEqual(len(refpix), 2)
            self.assertEqual(refpix[0], 0.5 * (imageparam.imsize[0] - 1))
            self.assertEqual(refpix[1], 0.5 * (imageparam.imsize[1] - 1))

            phasecenter_rec = imagewriter.parse_phasecenter(imageparam.phasecenter)
            center_x = qa.convert(phasecenter_rec['m0'], 'rad')['value']
            center_y = qa.convert(phasecenter_rec['m1'], 'rad')['value']
            #print center_x, center_y
            eps = 1.0e-11
            self.assertClose(refval[0], center_x, eps)
            self.assertClose(refval[1], center_y, eps)
            cellx = -qa.convert(qa.quantity(imageparam.cell[0]), 'rad')['value']
            celly = qa.convert(qa.quantity(imageparam.cell[1]), 'rad')['value']
            #print cellx, celly
            self.assertClose(incr[0], cellx, eps)
            self.assertClose(incr[1], celly, eps)

            # spectral coordinate
            specref = csys.referencecode(type='spectral')[0]
            self.assertEqual(specref, 'LSRK')
            refpix = self._get_reference(csys, 'referencepixel', 'spectral')
            refval = self._get_reference(csys, 'referencevalue', 'spectral')
            incr = self._get_reference(csys, 'increment', 'spectral')
            self.assertEqual(refpix, 0.0)
            #self.assertEqual(refval, qa.convert(start, 'Hz')['value'])
            #self.assertEqual(incr, qa.convert(width, 'Hz')['value'])
            self.assertEqual(refval, qa.convert(imageparam.start, 'Hz')['value'])
            self.assertEqual(incr, qa.convert(imageparam.width, 'Hz')['value'])

            # stokes coordinate
            refpix = self._get_reference(csys, 'referencepixel', 'stokes')
            refval = self._get_reference(csys, 'referencevalue', 'stokes')
            incr = self._get_reference(csys, 'increment', 'stokes')
            self.assertEqual(refpix, 0.0)
            self.assertEqual(refval, 1.0)
            self.assertEqual(incr, 1.0)

            # verify image meta data
            self.assertEqual(csys.observer(), imagemeta.observer)
            self.assertEqual(csys.telescope(), imagemeta.telescope)
            self.assertTrue(qa.eq(csys.restfrequency(), imagemeta.rest_frequency))
            epoch = csys.epoch()
            obsdate = imagemeta.observing_date
            self.assertEqual(epoch['refer'], obsdate['refer'])
            self.assertClose(epoch['m0']['value'], obsdate['m0']['value'])

            # image data
            chunk = ia.getchunk()
            imageshape = chunk.shape
            self.assertEqual(imageshape, expected_shape)

            shaped_chunk = chunk[:, :, :, 0]
            print(shaped_chunk.shape, arr.shape)
            self.assertTrue(numpy.allclose(arr, shaped_chunk),
                            msg='expected {0} actual {1} (maxdiff {2})'.format(arr,
                                                                               shaped_chunk,
                                                                               abs(arr - shaped_chunk).max()))

        finally:
            ia.done()

    def get_default_imageparam(self):
        qa = casa.CreateCasaQuantity()
        phasecenter = '9:00:00 -60.00.00 J2000'
        #phasecenter_rec = imagewriter.parse_phasecenter(phasecenter_str)
        start = qa.quantity('101GHz')
        width = qa.quantity('1MHz')
        nchan = 3
        imsize = [10, 9]
        cell = '1arcsec'
        imageparam = paramcontainer.ImageParamContainer(imagename=self.imagename,
                                                        phasecenter=phasecenter,
                                                        imsize=imsize,
                                                        cell=cell,
                                                        start=start,
                                                        width=width,
                                                        nchan=nchan,
                                                        outframe='LSRK',
                                                        stokes='I')

        return imageparam

    def make_image_array(self, imageparam):
        # image shape is (nx,ny,nstokes,nchan)
        imageshape = (imageparam.imsize[0],
                      imageparam.imsize[1],
                      imageparam.nchan,)

        vstart = 1.0
        vstep = 0.01
        vend = vstart + numpy.prod(imageshape) * vstep
        arr = numpy.arange(vstart, vend, vstep, dtype=numpy.float32)
        #print len(arr), imageshape, numpy.prod(imageshape)
        arr = arr.reshape(imageshape)

        return arr

    def test_default_meta(self):
        # default imageparam
        imageparam = self.get_default_imageparam()

        # create image array
        arr = self.make_image_array(imageparam)

        # perform test with default imagemeta (None)
        self._run_test(imageparam, arr, None)

    def test_custom_meta(self):
        me = casa.CreateCasaMeasure()

        # default imageparam
        imageparam = self.get_default_imageparam()

        # create image array
        arr = self.make_image_array(imageparam)

        # custom imagemeta
        imagemeta = paramcontainer.ImageMetaInfoContainer(observer='Takeshi Nakazato',
                                                          telescope='NRO',
                                                          observing_date=me.epoch('UTC', 'today'),
                                                          rest_frequency='101GHz')

        # perform test with default imagemeta (None)
        self._run_test(imageparam, arr, imagemeta)

    def test_coordsys(self):
        paramdict = {
            'imagename': 'test_coordsys.fits',
            'imsize': 100,
            'cell': '1arcsec',
            'projection': 'SIN',
            'phasecenter': '0:0:0 0.0.0 J2000',
            'nchan': 10,
            'start': '100GHz',
            'width': '10MHz',
            'outframe': 'LSRK',
            'stokes': 'I'
        }
        imparam = paramcontainer.ImageParamContainer.CreateContainer(**paramdict)
        print(imparam.imagename)
        self.assertEqual(imparam.imagename, paramdict['imagename'])

        imshape = (imparam.imsize[0], imparam.imsize[1],
                   imparam.nchan, 1,)
        imarray = numpy.zeros(imshape, dtype=numpy.float32)

        writer = imagewriter.ImageWriter(imageparam=imparam, imagearray=imarray)
        csys = writer._setup_coordsys()
        self.result = csys

    def test_phasecenter(self):
        phasecenter_str = '15:30:0 -10.30.00 J2000'
        phasecenter_rec = imagewriter.parse_phasecenter(phasecenter_str)
        self.result = phasecenter_rec
        lonstr, latstr, ref = phasecenter_str.split()
        qa = casa.CreateCasaQuantity()
        me = casa.CreateCasaMeasure()

        lon = qa.convert(qa.quantity(lonstr), 'rad')
        # to make value in ranges [-pi, pi]
        pi = qa.constants('pi')['value']
        lon['value'] = (lon['value'] + pi) % (2 * pi) - pi
        lat = qa.quantity(latstr)

        print(lon)
        print(lat)
        print(phasecenter_str)

        self.assertEqual(me.gettype(phasecenter_rec), 'Direction')
        self.assertEqual(me.getref(phasecenter_rec), ref)
        phasecenter_value = me.getvalue(phasecenter_rec)
        actual_lon = qa.convert(phasecenter_rec['m0'], 'rad')['value']
        expected_lon = qa.convert(lon, 'rad')['value']
        diff = abs((actual_lon - expected_lon) / expected_lon)
        eps = 1.0e-15
        self.assertLess(diff, eps)
        self.assertTrue(qa.eq(lat, phasecenter_value['m1']))


def suite():
    test_items = ['test_default_meta',
                  'test_default_meta',
                  'test_coordsys',
                  'test_phasecenter']
    test_suite = utils.generate_suite(ImageWriterTest,
                                      test_items)
    return test_suite

from __future__ import absolute_import

import unittest
import numpy
import os
import shutil

import almasparsemodeling.core as core
import almasparsemodeling.external.casa as casa
import almasparsemodeling.test.utils as utils

class VisibilityReaderTest(utils.TestBase):
    """
    Test suite for visreader
    
       test_iter: test iteration without data selection
       test_msselect: test msselection (select only spw 1)
       test_items: test record items to be returned
       test_iter_columns: test iteration customization using columns
       test_iter_interval: test iteration customization using interval
       test_iter_nrow: test iteration customization using nrow
       test_msselect_chan: test msselection with channel selection
       test_generator: test working set generator
    """
    result = None
    datapath = os.path.join(os.path.expanduser('~'), 
                            'casadevel/data/regression/unittest/listobs')
    vis = 'uid___X02_X3d737_X1_01_small.ms'

    def setUp(self):
        print 'setUp: copying {0}...'.format(self.vis)
        shutil.copytree(os.path.join(self.datapath, self.vis), self.vis)
        
        self.assertTrue(os.path.exists(self.vis))

        with casa.OpenTableForRead(os.path.join(self.vis, 'DATA_DESCRIPTION')) as tb:
            spw_ids = tb.getcol('SPECTRAL_WINDOW_ID')
            pol_ids = tb.getcol('POLARIZATION_ID')
        
        with casa.OpenTableForRead(os.path.join(self.vis, 'SPECTRAL_WINDOW')) as tb:
            num_chans = tb.getcol('NUM_CHAN')
            
        with casa.OpenTableForRead(os.path.join(self.vis, 'POLARIZATION')) as tb:
            num_corrs = tb.getcol('NUM_CORR')
            
        self.nchanmap = dict(((i, num_chans[spw_ids[i]],) for i in xrange(len(spw_ids))))
        self.ncorrmap = dict(((i, num_corrs[pol_ids[i]],) for i in xrange(len(pol_ids))))
    
    def tearDown(self):
        if os.path.exists(self.vis):
            print 'tearDown: removing {0}...'.format(self.vis)
            shutil.rmtree(self.vis)
            
        self.assertFalse(os.path.exists(self.vis))
            
    def _expected_nrows(self, additional_taql=None):
        taql = 'ANTENNA1 != ANTENNA2'
        if additional_taql is not None:
            taql = '&&'.join([taql, additional_taql])
            
        with casa.SelectTableForRead(self.vis, taql) as tb:
            expected = tb.nrows()
            
        return expected
    
    def verify_chunk_consistency(self, chunk, chunk_id):
        self.assertTrue('chunk_id' in chunk)
        self.assertEqual(chunk['chunk_id'], chunk_id)
        
    def test_iter(self):
        visparam = core.VisParamContainer(vis=self.vis)
        reader = core.VisibilityReader(visparam)
        
        nrow_total = 0
        chunk_id = 0
        for chunk in reader.readvis(items=['time', 'data_desc_id', 'data', 'weight']):
            nrow = chunk['time'].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk.values()[0].shape)
            
            self.verify_chunk_consistency(chunk, chunk_id)
            nrow_total += nrow
            chunk_id += 1
            
            data_list = chunk['data']
            ddid_list = chunk['data_desc_id']
            weight_list = chunk['weight']
            
            for irow in xrange(nrow):
                data = data_list[...,irow]
                weight = weight_list[...,irow]
                ddid = ddid_list[irow]
                nchan = self.nchanmap[ddid]
                ncorr = self.ncorrmap[ddid]
                self.assertEqual(data.shape, (ncorr, nchan,))
                self.assertEqual(weight.shape, (ncorr,))
        
        print 'LOG iterated {0} times in total'.format(chunk_id)
        print 'LOG total number of rows {0}'.format(nrow_total)
                
        nrow_total_expected = self._expected_nrows()
            
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
    
    def test_msselect(self):
        visparam = core.VisParamContainer(vis=self.vis, spw='1')
        reader = core.VisibilityReader(visparam)

        chunk_id = 0
        nrow_total = 0
        for chunk in reader.readvis(items=['data_desc_id']):
            nrow = chunk['data_desc_id'].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk['data_desc_id'].shape)
            
            self.verify_chunk_consistency(chunk, chunk_id)
            chunk_id += 1
            nrow_total += nrow
            
            self.assertTrue(chunk.has_key('data_desc_id'))
            dd = chunk['data_desc_id']
            self.assertTrue(numpy.all(dd == 1))
            
        nrow_total_expected = self._expected_nrows('DATA_DESC_ID==1')
            
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
        
    
    def test_items(self):
        visparam = core.VisParamContainer(vis=self.vis)
        reader = core.VisibilityReader(visparam)

        chunk_id = 0
        nrow_total = 0
        items = ['antenna1', 'antenna2', 'uvw', 'time']
        for chunk in reader.readvis(items=items):
            nrow = chunk[items[0]].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk.values()[0].shape)

            self.verify_chunk_consistency(chunk, chunk_id)
            chunk_id += 1
            nrow_total += nrow
            
            # chunk has additional field 'chunk_id'
            self.assertEqual(len(chunk), len(items) + 1)
            for item in items:
                self.assertTrue(chunk.has_key(item))
                        
        nrow_total_expected = self._expected_nrows()
            
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
            
    def test_iter_columns(self):
        visparam = core.VisParamContainer(vis=self.vis)
        reader = core.VisibilityReader(visparam)

        chunk_id = 0
        nrow_total = 0
        columns = ['ANTENNA1']
        items = ['antenna1']
        antenna_list = []
        for chunk in reader.readvis(items=items, columns=columns, adddefault=False):
            nrow = chunk[items[0]].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk[items[0]].shape)

            self.verify_chunk_consistency(chunk, chunk_id)
            chunk_id += 1
            nrow_total += nrow
            
            antenna = chunk['antenna1']
            self.assertTrue(numpy.all(antenna == antenna[0]))
            
            self.assertFalse(antenna[0] in antenna_list)
            
            antenna_list.append(antenna[0])
                        
        nrow_total_expected = self._expected_nrows()
            
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
    
    def test_iter_interval(self):
        visparam = core.VisParamContainer(vis=self.vis)
        reader = core.VisibilityReader(visparam)

        chunk_id = 0
        nrow_total = 0
        interval = 32
        for chunk in reader.readvis(items=['time'], 
                                    interval=interval):
            nrow = chunk['time'].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk['time'].shape)

            self.verify_chunk_consistency(chunk, chunk_id)
            chunk_id += 1
            nrow_total += nrow
            
            times = chunk['time']
            mintimeidx = times.argmin()
            maxtimeidx = times.argmax()
            timemin = times[mintimeidx]
            timemax = times[maxtimeidx]
            time_interval = timemax - timemin
            self.assertLessEqual(time_interval, interval)
                        
        nrow_total_expected = self._expected_nrows()
            
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
    
    def test_iter_nrow(self):
        visparam = core.VisParamContainer(vis=self.vis)
        reader = core.VisibilityReader(visparam)

        chunk_id = 0
        nrow_total = 0
        nrow_chunk = 4
        for chunk in reader.readvis(nrow=nrow_chunk):
            nrow = chunk['time'].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk.values()[0].shape)

            self.verify_chunk_consistency(chunk, chunk_id)
            chunk_id += 1
            nrow_total += nrow
            
            self.assertLessEqual(nrow, nrow_chunk)
            
        nrow_total_expected = self._expected_nrows()
           
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
    
    def test_msselect_chan(self):
        self.skipTest('Channel selection is not effective to ms iterator')
        visparam = core.VisParamContainer(vis=self.vis, spw='0:0~9')
        reader = core.VisibilityReader(visparam)

        chunk_id = 0
        nrow_total = 0
        for chunk in reader.readvis(items=['time', 'data_desc_id', 'data']):
            nrow = chunk['time'].shape[-1]
            print 'LOG chunk {0}: nrow={1} items={2} itemshape={3}'.format(
                chunk_id,
                nrow,
                chunk.keys(),
                chunk.values()[0].shape)

            self.verify_chunk_consistency(chunk, chunk_id)
            chunk_id += 1
            nrow_total += nrow
            
            data_list = chunk['data']
            ddid_list = chunk['data_desc_id']
            
            for irow in xrange(nrow):
                data = data_list[:,:,irow]
                ddid = ddid_list[irow]
                self.assertEqual(ddid, 0)
                nchan = min(self.nchanmap[ddid], 10)
                print 'nchan = {0}'.format(nchan)
                ncorr = self.ncorrmap[ddid]
                self.assertEqual(data.shape, (ncorr, nchan,))
        
        print 'LOG iterated {0} times in total'.format(chunk_id)
        print 'LOG total number of rows {0}'.format(nrow_total)
                
        nrow_total_expected = self._expected_nrows('DATA_DESC_ID==0')  
          
        print 'LOG number of rows for input ms {0}'.format(nrow_total_expected)
        self.assertEqual(nrow_total_expected, nrow_total)
    
def suite():
    test_items = ['test_iter',
                  'test_msselect',
                  'test_items',
                  'test_iter_columns',
                  'test_iter_interval',
                  'test_iter_nrow',
                  'test_msselect_chan']
    test_suite = utils.generate_suite(VisibilityReaderTest, 
                                      test_items)
    return test_suite

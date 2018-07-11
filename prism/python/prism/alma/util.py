from __future__ import absolute_import

import os
import numpy

import prism.external.casa as casa
import prism.core.util as core_util

RandomIndexGenerator = core_util.RandomIndexGenerator
    
class ImageConfigurationHelper(object):
    @staticmethod
    def get_antenna_diameter(vis):
        """
        Read antenna diameter (DISH_DIAMETER)
        
        Return antenna diameter in m
        """
        assert os.path.exists(vis)
        antenna_table = os.path.join(vis, 'ANTENNA')
        assert os.path.exists(antenna_table)
        
        with casa.OpenTableForRead(antenna_table) as tb:
            antenna_diameter = tb.getcol('DISH_DIAMETER')
            
        return antenna_diameter
    
    @staticmethod
    def get_observing_frequency(vis):
        """
        Read observing frequency (REF_FREQUENCY) 
        
        Return frequency value in Hz as a dictionary of 
        {data_desc_id: frequency}
        """
        assert os.path.exists(vis)
        spectralwindow_table = os.path.join(vis, 'SPECTRAL_WINDOW')
        assert os.path.exists(spectralwindow_table)
        
        with casa.OpenTableForRead(spectralwindow_table) as tb:
            ref_frequency = tb.getcol('REF_FREQUENCY')
        
        datadescription_table = os.path.join(vis, 'DATA_DESCRIPTION')
        assert os.path.exists(datadescription_table)
        
        with casa.OpenTableForRead(datadescription_table) as tb:
            ddid_map = tb.getcol('SPECTRAL_WINDOW_ID')
            
        observing_frequency = {}
        for (ddid, spwid) in enumerate(ddid_map):
            if spwid >= 0 and spwid < len(ref_frequency):
                observing_frequency[ddid] = ref_frequency[spwid]
        
        return observing_frequency
    
    @staticmethod
    def calc_primary_beam(antenna_diameter, frequency):
        """
        calculate antenna primary beam size 
        
        Inputs:
            antenna_diameter -- antenna diameter in m
            frequency -- observing frequency in GHz
        
        Returns:
            primary beam size in arcsec
        """
        factor = 1.13
        beamsize_in_arcsec = factor * 6.188e4 / antenna_diameter / frequency
        
        return beamsize_in_arcsec
        
        
    @staticmethod
    def suggest_imaging_param(visparam):
        vis = visparam.vis
        msselect = visparam.as_msselection()
        
        with casa.OpenMS(vis) as ms:
            ms.msselect(msselect, onlyparse=False)
            data = ms.getdata(['uvw', 'data_desc_id', 'antenna1', 'antenna2'])
        uvw = data['uvw']
        ddid = data['data_desc_id']
        antenna1 = data['antenna1']
        antenna2 = data['antenna2']
        
        observing_frequency = ImageConfigurationHelper.get_observing_frequency(vis)
        antenna_diameter = ImageConfigurationHelper.get_antenna_diameter(vis)
        
        # maximum antenna primary beam size [arcsec]
        min_freq = min(observing_frequency.values()) * 1e-9 # Hz -> GHz
        min_diameter = min(antenna_diameter)
        primary_beam = ImageConfigurationHelper.calc_primary_beam(min_diameter,
                                                                  min_freq)     
        
        qa = casa.CreateCasaQuantity()
        c = qa.convert(qa.constants('c'), 'm/s')['value']
        nrow = len(ddid)
        umax = 0.0
        vmax = 0.0
        for irow in range(nrow):
            f = observing_frequency[ddid[irow]]
            u = uvw[0,irow] / c * f
            v = uvw[1,irow] / c * f
            umax = max(umax, u)
            vmax = max(vmax, v)
            
        rad2arcsec = 180.0 / numpy.pi * 3600.0
        dl = 1.0 / (2 * umax) * rad2arcsec # rad -> arcsec
        dm = 1.0 / (2 * vmax) * rad2arcsec # rad -> arcsec
        
        M = int(numpy.ceil(primary_beam / dl)) + 12
        N = int(numpy.ceil(primary_beam / dm)) + 12
        
        suggested = {'cell': ['{}arcsec'.format(dl), '{}arcsec'.format(dm)],
                     'imsize': [M, N]}
            
        return suggested
    
    
from __future__ import absolute_import

#import collections

import prism.external.casa as casa

import prism.core.paramcontainer as base_container
import prism.core.datacontainer as datacontainer
                

class VisParamContainer(base_container.ParamContainer):
    def __init__(self, vis='', field='', spw='', timerange='', uvrange='',
                 antenna='', scan='', observation='', intent='', array='',
                 polarization='', taql='',
                 datacolumn='corrected'):
        """
        Constructor
        """
        self.InitContainer(locals())
        
    @property
    def datacolumn(self):
        return getattr(self, '_datacolumn', None)
    
    @datacolumn.setter
    def datacolumn(self, value):
        if value not in ('data', 'corrected',):
            raise ValueError('datacolumn should be either \'data\' or \'corrected\'')
        else:
            self._datacolumn = value
        
    @property    
    def antenna(self):
        if self._antenna == '':
            # default is select all cross-correlation
            return '*&*'
        else:
            return self._antenna
    
    @antenna.setter
    def antenna(self, value):
        self._antenna = value
        
    def as_msselection(self):
        """
        Return a dictionary that can be passed to ms.msselect directly.
        """
        return {'field': self.field,
                'spw': self.spw,
                'time': self.timerange,
                'uvdist': self.uvrange,
                'baseline': self.antenna,
                'scan': self.scan,
                'observation': self.observation,
                'scanintent': self.intent,
                'polarization': self.polarization,
                'array': self.array,
                'taql': self.taql}
        
    def as_msindex(self):
        """
        Return a dictionary of selected indices.
        """
        with casa.OpenMS(self.vis) as ms:
            status = ms.msselect(self.as_msselection(), onlyparse=True)
            idx = ms.msselectedindices()
        return idx

class ImageParamContainer(base_container.ParamContainer):
    """
    Parameter container for imaging
    
    start, width, and nchan are defined as follows:
        
          start=<center frequency of first image channel>
            |
        |-------|-------|-------| nchan=3
        |<----->|
          width=<constant channel width of image channel>
    """     
    def __init__(self, imagename='', imsize=100, cell='1arcsec', projection='SIN', 
                 phasecenter='', nchan=-1, start='', width='', outframe='LSRK', 
                 stokes='I'):
        """
        Constructor
        """
        self.InitContainer(locals())
    
    @property
    def imsize(self):
        return getattr(self, '_imsize', None)
    
    @imsize.setter
    def imsize(self, value):
        if hasattr(value, '__iter__'):
            self._imsize = list(value)
        else:
            self._imsize = [int(value)]
            
        if len(self._imsize) == 0:
            raise TypeError('given imsize is not correct')
        elif len(self._imsize) == 1:
            self._imsize = [self._imsize[0], self._imsize[0]]
        else:
            self._imsize = self._imsize[:2]
    
    @property
    def cell(self):
        return getattr(self, '_cell', None)
    
    @cell.setter
    def cell(self, value):
        if hasattr(value, '__iter__'):
            self._cell = list(value)
        else:
            self._cell = [str(value)]
            
        if len(self._cell) == 0:    
            raise TypeError('given cell is not correct')
        elif len(self._cell) == 1:
            self._cell = [self._cell[0], self._cell[0]]
        else:
            self._cell = self._cell[:2]
            
    @property
    def start(self):
        return getattr(self, '_start', None)
    
    @start.setter
    def start(self, value):
        qa = casa.CreateCasaQuantity()
        if isinstance(value, str):
            self._start = qa.quantity(value)
        elif isinstance(value, dict):
            self._start = value
        else:
            # should be numeric value that 
            # are expected to be channel
            self._start = value
            
    @property
    def width(self):
        return getattr(self, '_width', None)
    
    @width.setter
    def width(self, value):
        qa = casa.CreateCasaQuantity()
        if isinstance(value, str):
            self._width = qa.quantity(value)
        elif isinstance(value, dict):
            self._width = value
        else:
            # should be numeric value that 
            # are expected to be channel
            self._width = value
        
    @property    
    def uvgridconfig(self):
        if not hasattr(self, '_uvgridconfig'):
            qa = casa.CreateCasaQuantity()
            cellx = qa.quantity(self.cell[0])
            celly = qa.quantity(self.cell[1])
            nu = self.imsize[0]
            nv = self.imsize[1]
            wx= nu * qa.convert(cellx, 'rad')['value']
            wy = nv * qa.convert(celly, 'rad')['value']
            cellu = 1 / wx
            cellv = 1 / wy
            # offset must always be pixel center even if nu and/or nv 
            # is even (which causes offset value to be non-integer)
            offsetu = int(nu) // 2 # make sure integer operation
            offsetv = int(nv) // 2 # make sure integer operation
#             UVGridConfig = collections.namedtuple('UVGridCondig', 
#                                                   ['cellu', 'cellv',
#                                                    'nu', 'nv',
#                                                    'offsetu', 'offsetv'])
            self._uvgridconfig = datacontainer.UVGridConfig(cellu=cellu,
                                                            cellv=cellv,
                                                            nu=nu,
                                                            nv=nv,
                                                            offsetu=offsetu,
                                                            offsetv=offsetv)
        return self._uvgridconfig

            
class ImageMetaInfoContainer(base_container.ParamContainer):
    @staticmethod
    def fromvis(vis):
        """
        Construct ImageMetaInfoContainer instance from visibility data (MS).
        """
        with casa.OpenMSMetaData(vis) as msmd:
            observers = msmd.observers()
            observatories = msmd.observatorynames()
            observingdate = msmd.timerangeforobs(0)
            position = msmd.observatoryposition(0)
            restfreqs = msmd.restfreqs()
        return ImageMetaInfoContainer(observer=observers[0], 
                                      telescope=observatories[0],
                                      telescope_position=position,
                                      observing_date=observingdate['begin'])
    
    def __init__(self, observer='', observing_date='', telescope='ALMA', telescope_position=None, 
                 rest_frequency=''):
        self.InitContainer(locals())
        
    @property
    def observer(self):
        return getattr(self, '_observer', None)
    
    @observer.setter
    def observer(self, value):
        if not isinstance(value, str) or len(value) == 0:
            self._observer = 'Sherlock Holmes'
        else:
            self._observer = value
        
    @property
    def telescope_position(self):
        return getattr(self, '_telescope_position', None)
    
    @telescope_position.setter
    def telescope_position(self, value):
        me = casa.CreateCasaMeasure()
        if value is None:
            # query to CASA database
            self._telescope_position = me.observatory(self.telescope)
            if len(self._telescope_position) == 0:
                raise ValueError('telescope "{0}" is unknown. you have to specify telescope position explicitly.'.format(self.telescope))
        elif isinstance(value, dict) and me.ismeasure(value):
            self._telescope_position = value
        else:
            raise ValueError('Invalid telescope position: {0}'.format(value))
    
    @property
    def observing_date(self):
        return getattr(self, '_observing_date', None)
    
    @observing_date.setter
    def observing_date(self, value):
        me = casa.CreateCasaMeasure()
        if isinstance(value, dict) and me.ismeasure(value):
            self._observing_date = value
        else:
            self._observing_date = me.epoch('UTC', 'today')
            
    @property
    def rest_frequency(self):
        return getattr(self, '_rest_frequency', None)
    
    @rest_frequency.setter
    def rest_frequency(self, value):
        qa = casa.CreateCasaQuantity()
        if isinstance(value, str):
            if len(value) > 0:
                self._rest_frequency = qa.quantity(value)
            else:
                self._rest_frequency = qa.quantity(1.0e9, 'Hz')
        elif isinstance(value, dict):
            self._rest_frequency = value
        else:
            # should be numeric
            self._rest_frequency = qa.quantity(value, 'Hz')
            
class GridParamContainer(base_container.ParamContainer):
    def __init__(self, convsupport=3, convsampling=100, gridfunction=None):
        self.InitContainer(locals()) 
    
    

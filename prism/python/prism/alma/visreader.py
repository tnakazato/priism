from __future__ import absolute_import

from . import paramcontainer
from . import visconverter
import prism.external.casa as casa

class VisibilityReader(object):
    def __method_name(self, basename):
        if casa.casa_version.major == 5:
            if casa.casa_version.minor >= 3:
                return basename
            elif casa.casa_version.minor >= 1:
                return '{}2'.format(basename)
        raise RuntimeError('Unsupported CASA version {}.{}.{}-{}'.format(casa.casa_version.major,
                                                                         casa.casa_version.minor,
                                                                         casa.casa_version.major,
                                                                         casa.casa_version.build))       
    
    @property
    def iterinit(self):
        return self.__method_name('iterinit')
    
    @property
    def iterorigin(self):
        return self.__method_name('iterorigin')
    
    @property
    def getdata(self):
        return self.__method_name('getdata')
    
    @property
    def iternext(self):
        return self.__method_name('iternext') 
    
    """
    VisibilityReader is responsible for scanning and reading MS 
    according to user-specified data selection (MS name, 
    fields, spws, etc.).
    
    It has a generator method "readvis" to provide an 
    access to all selected data.
    """
    def __init__(self, visparam):
        """
        Constructor
        
        visparam -- visibility data selection parameter 
                    as an instance of VisParamContainer
        """
        self.visparam = visparam
        self.sel = None
                
        assert isinstance(self.visparam, paramcontainer.VisParamContainer)
                                        
    def readvis(self, items=visconverter.VisibilityConverter.required_columns, 
                columns=[], interval=0.0, nrow=0, adddefault=True):
        """
        read visibility data accoding to the data selection 
        provided to constructor.
        
        columns -- iteration axis
        interval -- time interval to group together
        nrow -- number of row for returned data chunk
        adddefault -- add default sort columns to iteration axis
        """
        vis = self.visparam.vis
        msselect = self.visparam.as_msselection()

        with casa.OpenMS(vis) as ms:
            ms.msselect(msselect, onlyparse=False)
            
            # method names depend on CASA version
            ms_iterinit = getattr(ms, self.iterinit)
            ms_iterorigin = getattr(ms, self.iterorigin)
            ms_getdata = getattr(ms, self.getdata)
            ms_iternext = getattr(ms, self.iternext)
        
            # not using new iterator as it doesn't support 
            # reading meta information such as UVW...
            # iterate through MS using VI/VB2 framework
            ms_iterinit(columns, interval, nrow, adddefault)
            more_chunks = ms_iterorigin()
            chunk_id = 0
            while (more_chunks):
                rec = ms_getdata(items)
                rec['chunk_id'] = chunk_id
                print 'LOG: read visibility chunk #{0}'.format(chunk_id)
                yield rec
                more_chunks = ms_iternext()
                chunk_id += 1

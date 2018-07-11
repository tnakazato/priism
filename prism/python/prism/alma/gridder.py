from __future__ import absolute_import

import numpy

import prism.core.util as util
import prism.core.paramcontainer as paramcontainer
import prism.core.datacontainer as datacontainer
import prism.external.casa as casa
import prism.external.sakura as sakura

class GridderWorkingSet(paramcontainer.ParamContainer):
    """
    Working set for gridder
    
    NOTE: flag=True indicates *VALID* data
          flag=False indicates *INVALID* data
    
    u, v --- position in uv-plane (nrow)
    rdata --- real part of visibility data (nrow, npol, nchan)
    idata --- imaginary part of visibility data (nrow, npol, nchan)
    flag --- channelized flag (nrow, npol, nchan)
    weight --- visibility weight (nrow, nchan) 
    row_flag --- row flag (nrow)
    channel_map --- channel mapping between raw visibility
                    and gridded visibility (nchan)
    pol_map --- polarization mapping between raw visibility
                and gridded visibility (npol)
    """
    def __init__(self, data_id=None, u=0.0, v=0.0, rdata=None, idata=None, 
                 flag=None, weight=None, row_flag=None, channel_map=None,
                 pol_map=None):
        self.InitContainer(locals())
        
    def __from_shape(self, axis=0):
        if self.rdata is None:
            return 0
        else:
            # should be numpy array
            shape = self.rdata.shape
            if len(shape) < axis + 1:
                return 1
            else:
                return shape[axis]    
    
    @property
    def nrow(self):
        return self.__from_shape(axis=0)
    
    @property
    def nchan(self):
        return self.__from_shape(axis=2)
    
    @property
    def npol(self):
        return self.__from_shape(axis=1)
    
    @property
    def start(self):
        return 0
    
    @property
    def end(self):
        return self.nrow - 1
    
    @property
    def data_id(self):
        if hasattr(self, '_data_id'):
            return self._data_id
        else:
            raise ValueError('invalid data_id: Undefined')
        
    @data_id.setter
    def data_id(self, value):
        if not isinstance(value, int):
            raise ValueError('invalid data_id ({0}). Should be int'.format(value))
        else:
            self._data_id = value
            
    @property
    def pol_map(self):
        if hasattr(self, '_pol_map'):
            return self._pol_map
        else:
            self._pol_map = sakura.empty_aligned((self.npol,), dtype=numpy.int32)
            self._pol_map[:] = numpy.arange(self.npol)
            return self._pol_map
    
    @pol_map.setter
    def pol_map(self, value):
        if value is None:
            #self._pol_map = sakura.empty_aligned((self.npol,), dtype=numpy.int32)
            #self._pol_map[:] = range(self.npol)
            pass
        else:
            try:
                if len(value) == self.npol and isinstance(value[0], int):
                    self._pol_map = value
                else:
                    raise
            except Exception, e:
                raise ValueError('invalid pol_map ({0}). Should be int list or None.'.format(value))
        
                
class GridFunctionUtil(object):
    @staticmethod
    def allocate(convsupport, convsampling, init=False):
        n = (convsupport + 1) * convsampling * 2 
        gf = sakura.empty_aligned((n,), dtype=numpy.float32)
        if init:
            gf[:] = 0.0
        return gf
        
    @staticmethod
    def box(convsupport, convsampling):
        """
        Generate Box gridding kernel whose value is 1.0 inside 
        convsupport pixel while 0.0 otherwise.
        
        convsupport -- support radius in pixel
        convsampling -- number of sampling per pixel
        """
        gf = GridFunctionUtil.allocate(convsupport, convsampling)
        gf[:convsampling] = 1.0
        gf[convsampling:] = 0.0
        return gf
    
    @staticmethod
    def gauss(convsupport, convsampling, hwhm):
        """
        Generate Gaussian gridding kernel
        
        convsupport -- support radius in pixel
        convsampling -- number of sampling per pixel
        hwhm -- Half-Width of Half-Maximum in pixel unit
        """
        gf = GridFunctionUtil.allocate(convsupport, convsampling)
        gf[:] = 0.0
        sigma = float(hwhm) / numpy.sqrt(2.0 * numpy.log(2.0))
        m = convsupport * convsampling
        for i in xrange(m):
            x = float(i) / float(convsampling)
            gf[i] = numpy.exp(-(x * x) / (2.0 * sigma * sigma))
        return gf
    
    @staticmethod
    def sf(convsupport, convsampling):
        """
        Generate prolate-Spheroidal gridding kernel
        
        convsupport -- support radius in pixel
        convsampling -- number of sampling per pixel
        """
        gf = GridFunctionUtil.allocate(convsupport, convsampling)
        m = convsupport * convsampling
        for i in xrange(m):
            nu = float(i) / float(m)
            val = GridFunctionUtil.grdsf(nu)
            gf[i] = (1.0 - nu * nu) * val
        gf[m:] = 0.0
        # normalize so peak is 1.0
        gf *= 1.0 / gf[0]
        return gf 
    
    @staticmethod
    def grdsf(nu):
        """
        cf. casacore/scimath_f/grdsf.f
        """
        P0 = [8.203343e-2, -3.644705e-1, 6.278660e-1,
             -5.335581e-1, 2.312756e-1]
        P1 = [4.028559e-3, -3.697768e-2, 1.021332e-1,
             -1.201436e-1, 6.412774e-2]
        Q0 = [1.0000000e0, 8.212018e-1, 2.078043e-1]
        Q1 = [1.0000000e0, 9.599102e-1, 2.918724e-1]
        nP = 4
        nQ = 2
        
        val = 0.0
        if 0.0 <= nu and nu < 0.75:
            P = P0
            Q = Q0
            nuend = 0.75
        elif 0.75 <= nu and nu <= 1.0:
            P = P1
            Q = Q1
            nuend = 1.0
        else:
            val = 0.0
            return val
        
        top = P[0]
        delnusq = nu * nu - nuend * nuend
        kdelnusq = 1.0
        for k in xrange(1, nP+1):
            kdelnusq *= delnusq
            top += P[k] * kdelnusq
            
        bot = Q[0]
        kdelnusq = 1.0
        for k in xrange(1, nQ+1):
            kdelnusq *= delnusq
            bot += Q[k] * kdelnusq
            
        if bot != 0.0:
            val = top / bot
        else:
            val = 0.0
        
        return val
    
class GridderResult(object):
    """
    Class to hold gridder result
    
    expected array shape for grid_real and grid_imag is 
    (nv, nu, npol, nchan)
    """
    def __init__(self, grid_real, grid_imag, wgrid_real, wgrid_imag=None, num_ws=None):
        self.real = grid_real
        self.imag = grid_imag
        self.wreal = wgrid_real
        self.wimag = wgrid_imag if wgrid_imag is not None else wgrid_real
        assert self.real.shape == self.imag.shape
        assert self.real.shape == self.wreal.shape
        assert self.real.shape == self.wimag.shape
        self.shape = self.real.shape
        
        # number of ws that are accumulated onto grid
        self.num_ws = num_ws if num_ws is not None else 0
    
class VisibilityGridder(object):
    """
    Configure grid and accumulate data onto each grid position.
    Data should be provided in the form of GridderWorkingSet 
    instance.
    """
    def __init__(self, gridparam, imageparam):
        self.gridparam = gridparam
        #self.visparam = visparam
        self.imageparam = imageparam
        self.num_ws = 0
        self._init()
        
    @property
    def convsupport(self):
        if hasattr(self, 'gridparam'):
            return self.gridparam.convsupport
        else:
            return None
        
    @property
    def convsampling(self):
        if hasattr(self, 'gridparam'):
            return self.gridparam.convsampling
        else:
            return None
        
    @property
    def weight_only(self):
        return False
    
    @property
    def nchan(self):
        if hasattr(self, 'imageparam'):
            nchan = self.imageparam.nchan
            if nchan > 0:
                return nchan
            else:
                return 1
        else:
            return 1
        
    @property
    def nkernel(self):
        if hasattr(self, 'gridparam'):
            return len(self.gridparam.gridfunction)
        else:
            return 0
        
    @property
    def gridfunction(self):
        if hasattr(self, 'gridparam'):
            return self.gridparam.gridfunction
        else:
            return None
    
    def _init(self):
        # grid parameter from visibility selection parameter
#         sel = self.visparam.as_msindex()
#         poldd = sel['poldd']
#         with casa.OpenTableForRead(os.path.join(self.visparam.vis,
#                                                 'DATA_DESCRIPTION')) as tb:
#             tsel = tb.selectrows(poldd)
#             polids = numpy.unique(tsel.getcol('POLARIZATION_ID'))
#             tsel.close()
#         with casa.OpenTableForRead(os.path.join(self.visparam.vis,
#                                                 'POLARIZATION')) as tb:
#             tsel = tb.selectrows(polids)
#             num_corrs = numpy.unique(tsel.getcol('NUM_CORR'))
#             tsel.close()
#             
#         assert len(num_corrs) == 1
#         self.npol = num_corrs[0]
        # so far npol should be 1 (only I)
        self.npol = 1
        
        # grid parameter from image parameter
        uvgridconfig = self.imageparam.uvgridconfig
        self.cellu = uvgridconfig.cellu
        self.cellv = uvgridconfig.cellv
        self.nu = uvgridconfig.nu
        self.nv = uvgridconfig.nv
        self.offsetu = uvgridconfig.offsetu
        self.offsetv = uvgridconfig.offsetv
        
        # create storage
        # require margin based on convsupport parameter since 
        # sakura gridding function ignore convsupport-pixels 
        # from spatial edges
        wsshape = (self.npol, self.nchan)
        self.wsum_real = sakura.empty_aligned(wsshape, dtype=numpy.float64)
        self.wsum_imag = sakura.empty_aligned(wsshape, dtype=numpy.float64)
        gridshape = (self.nv + 2 * self.convsupport, 
                     self.nu + 2 * self.convsupport, 
                     self.npol, self.nchan)
        self.grid_real = sakura.empty_aligned(gridshape, dtype=numpy.float32)
        self.grid_imag = sakura.empty_aligned(gridshape, dtype=numpy.float32)
        self.wgrid_real = sakura.empty_aligned(gridshape, dtype=numpy.float32)
        self.wgrid_imag = sakura.empty_aligned(gridshape, dtype=numpy.float32)
        
        # zero clear
        self._clear_grid()
        
        # number of ws to be gridded
        self.num_ws = 0
        
    def _clear_grid(self):
        self.grid_real[:] = 0
        self.grid_imag[:] = 0
        self.wgrid_real[:] = 0
        self.wgrid_imag[:] = 0
        self.wsum_real[:] = 0
        self.wsum_imag[:] = 0

    def grid(self, ws):
        """
        Accumulate data provided as working set onto grid.
        """
        print('LOG: accumulate visibility chunk #{0} onto grid'.format(ws.data_id))
        # shift uv (pixel) coordinates by convsupport to take into 
        # account margin pixels
        ws.u += self.convsupport
        ws.v += self.convsupport
        tmpu = ws.u.copy()
        tmpv = ws.v.copy()
        tmpd = ws.idata.copy()
        try:
            # grid ws as it is
            sakura.grid(ws, self.gridfunction,
                        self.convsupport, self.convsampling,
                        self.weight_only, 
                        self.grid_real, self.grid_imag,
                        self.wgrid_real, self.wgrid_imag,
                        self.wsum_real, self.wsum_imag)
            
            # then grid complex conjugate of ws
            ws.u[:] = 2 * self.offsetu - (ws.u - self.convsupport) + self.convsupport
            ws.v[:] = 2 * self.offsetv - (ws.v - self.convsupport) + self.convsupport
            ws.idata *= -1.0
            sakura.grid(ws, self.gridfunction,
                        self.convsupport, self.convsampling,
                        self.weight_only, 
                        self.grid_real, self.grid_imag,
                        self.wgrid_real, self.wgrid_imag,
                        self.wsum_real, self.wsum_imag)
        finally:
            # restore orignal uv (pixel) coordinates
            ws.u[:] = tmpu
            ws.v[:] = tmpv
            ws.idata[:] = tmpd
            ws.u -= self.convsupport
            ws.v -= self.convsupport
            self.num_ws += 1
        
    def get_result(self):
        # remove margin from grid array
        grid_shape = self.grid_real.shape
        outgrid_shape = (grid_shape[0] - 2 * self.convsupport, 
                         grid_shape[1] - 2 * self.convsupport,
                         grid_shape[2], grid_shape[3])
        grid_real  =  self.grid_real[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        grid_imag  =  self.grid_imag[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        wgrid_real = self.wgrid_real[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        wgrid_imag = self.wgrid_imag[self.convsupport:-self.convsupport, self.convsupport:-self.convsupport, :, :]
        assert grid_real.shape == outgrid_shape
        assert grid_imag.shape == outgrid_shape
        assert wgrid_real.shape == outgrid_shape
        assert wgrid_imag.shape == outgrid_shape
        
        nonzero_real = numpy.where(wgrid_real != 0.0)
        nonzero_imag = numpy.where(wgrid_imag != 0.0)
        uvgrid_real = sakura.empty_aligned(outgrid_shape, dtype=numpy.float64)
        uvgrid_imag = sakura.empty_aligned(outgrid_shape, dtype=numpy.float64)
        uvgrid_wreal = sakura.empty_aligned(outgrid_shape, dtype=numpy.float64)
        uvgrid_wreal[:] = wgrid_real
        if numpy.all(wgrid_real == wgrid_imag):
            uvgrid_wimag = None
        else:
            uvgrid_wimag = sakura.empty_aligned(outgrid_shape, dtype=numpy.float64)
            uvgrid_wimag[:] = wgrid_imag
        uvgrid_real[:] = 0.0
        uvgrid_imag[:] = 0.0
        uvgrid_real[nonzero_real] = grid_real[nonzero_real] / wgrid_real[nonzero_real]
        uvgrid_imag[nonzero_imag] = grid_imag[nonzero_imag] / wgrid_imag[nonzero_imag]
        
        result = datacontainer.GriddedVisibilityStorage(uvgrid_real, uvgrid_imag, 
                                                        uvgrid_wreal, uvgrid_wimag,
                                                        self.num_ws)
        return result
    
class CrossValidationVisibilityGridder(VisibilityGridder):
    """
    Gridder for cross validation.
    To separate visibility subset for cross validation, 
    this class has cache mechanism that separate a certain 
    amount of raw visibility from gridding and store them 
    for cross validation.
    """
    def __init__(self, gridparam, imageparam, num_ws, num_fold=10):
        super(CrossValidationVisibilityGridder, self).__init__(gridparam, imageparam)
        self.num_ws = num_ws
        self.num_fold = num_fold
        
        self.visibility_cache = []
        
        self.index_generator = util.VisibilitySubsetGenerator(self.num_ws, self.num_fold)
        
    def grid(self, ws, subset_id):
        """
        Separate ws if it is judged as a visibility cache for cross validation.
        Otherwise, accumulate ws onto grid.
        """
        subset_index = self.index_generator.get_subset_index(subset_id)
        #if ws.data_id % num_fold == subset_id:
        if ws.data_id in subset_index:
            self.visibility_cache.append(ws)
        else:
            super(CrossValidationVisibilityGridder).grid(ws)
            
    def get_visibility_cache(self):
        return self.visibility_cache
from __future__ import absolute_import

import os
import numpy
import ctypes

class CTypesUtilMixIn(object):
    def as_carray(self, attr):
        array = getattr(self, attr)
        return numpy.ctypeslib.as_ctypes(array)
    
    
def exec_line(f, varname):
    line = f.readline()
    exec(line.rstrip('\n'))
    val = locals()[varname]
    #print '{0} = {1}'.format(varname, val)
    return val
    
def shift_uvindex(n, iarr, inplace=True):
    """
    Assuming that input array index, iarr, is configured so that 
    zero-frequency term comes to the center, shift_uvindex shifts 
    iarr so that zero-frequency term comes to the first element. 
    It corresponds to numpy.fft.ifftshift.
    
    if n is odd:
        (a,b,c,d,e,f,g) -> (d,e,f,g,a,b,c)
    elif n is even:
        (a,b,c,d,e,f)   -> (d,e,f,a,b,c)
        
    n --- number of pixels along the axis
    iarr --- input array index
    inplace --- if True, iarr is edited instead to prepare output array
    """
    shift_term = n/2
    if inplace:
        ret = iarr
    else:
        ret = numpy.zeros_like(iarr)
    ret = (iarr + shift_term) % n
    return ret

def rshift_uvindex(n, iarr, inplace=True):
    """
    Assuming that input array index, iarr, is configured so that 
    zero-frequency term comes to the first element, rshift_uvindex 
    shifts iarr so that zero-frequency term comes to the center. 
    It corresponds to numpy.fft.fftshift.
    
    if n is odd:
        (a,b,c,d,e,f,g) -> (e,f,g,a,b,c,d)
    elif n is even:
        (a,b,c,d,e,f)   -> (d,e,f,a,b,c)

    n --- number of pixels along the axis
    iarr --- input array index
    inplace --- if True, iarr is edited instead to prepare output array
    """
    shift_term = n/2 + (n % 2)
    if inplace:
        ret = iarr
    else:
        ret = numpy.zeros_like(iarr)
    ret = (iarr + shift_term) % n
    return ret
    
    

class SparseImagingInputs(CTypesUtilMixIn):
    """
    Container for sparseimaging inputs
    """
    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            # read M
            M = exec_line(f, 'M')
            
            # read NX
            NX = exec_line(f, 'NX')
            
            # read NY
            NY = exec_line(f, 'NY')
            
            # skip headers
            f.readline()
            f.readline()
            f.readline()
            
            # read input data
            u = numpy.empty(M, dtype=numpy.int32)
            v = numpy.empty_like(u)
            yreal = numpy.empty(M, dtype=numpy.double)
            yimag = numpy.empty_like(yreal)
            noise = numpy.empty_like(yreal)
            for i in xrange(M):
                line = f.readline()
                values = line.split(',')
                u[i] = numpy.int32(values[0].strip())
                v[i] = numpy.int32(values[1].strip())
                yreal[i] = numpy.double(values[2].strip())
                yimag[i] = numpy.double(values[3].strip())
                noise[i] = numpy.double(values[4].strip())
                #print '{0} {1} {2} {3}'.format(u[i], v[i], yreal[i], yimag[i], noise[i])
                
            inputs = SparseImagingInputs(filename, M, NX, NY, u, v, yreal, yimag, noise)
            return inputs

    @staticmethod
    def from_gridder_result(gridder_result):
        """
        Convert GridderResult object into SparseImagingInputs object.
        uv-coordinate value is flipped for FFTW.
        """
        # infile is nominal value 
        infile = 'gridder_result'
        
        # m is the number of nonzero pixels
        nonzeros = numpy.nonzero(gridder_result.wreal)
        m = len(nonzeros[0])
        
        # numpy.nonzero returns index as numpy.int64 
        # however, libmfista_fft requires index in 32bit integer
        # value check is performed here
        iint32 = numpy.iinfo(numpy.int32)
        if nonzeros[0].max() > iint32.max or nonzeros[0].min() < iint32.min:
            raise ValueError('Pixel index along V-axis exceeded int32 limit')
        if nonzeros[1].max() > iint32.max or nonzeros[1].min() < iint32.min:
            raise ValueError('Pixel index along U-axis exceeded int32 limit')
        
        # nx, ny
        grid_shape = gridder_result.shape
        nv, nu, npol, nchan = grid_shape
        assert npol == 1
        assert nchan == 1
        nx = nu
        ny = nv
        
        # TODO: u, v must be flipped
        # flip u, v (grid indices) instead of visibility value
        # cast 64bit integer to 32bit integer
        unflipped_v = numpy.asarray(nonzeros[0], dtype=numpy.int32)
        unflipped_u = numpy.asarray(nonzeros[1], dtype=numpy.int32)
        u = shift_uvindex(nu, unflipped_u)
        v = shift_uvindex(nv, unflipped_v)
        
        # yreal, yimag are nonzero gridded visibility
        yreal = gridder_result.real[nonzeros]
        yimag = gridder_result.imag[nonzeros]
        
        # 20171102 suggestion by Ikeda-san
        # change sign according to pixel coordinate
        for i in xrange(len(yreal)):
            j = nonzeros[0][i]
            k = nonzeros[1][i]
            factor = (-1)**(j+k)
            yreal[i] *= factor
            yimag[i] *= factor
        
        # noise is formed as 1 / sqrt(weight)
        noise = gridder_result.wreal[nonzeros]
        noise = 1.0 / numpy.sqrt(noise)
        
        return SparseImagingInputs(infile, m, nx, ny, u, v, yreal, yimag, noise)
    
    def __init__(self, infile, M, NX, NY, u, v, yreal, yimag, noise):
        self.infile = infile
        self.m = M
        self.nx = NX
        self.ny = NY 
        self.u = u
        self.v = v
        self.yreal = yreal
        self.yimag = yimag
        self.noise = noise
        
    def export(self, filename):
        with open(filename, 'w') as f:
            print >> f, 'M = {0}'.format(self.m)
            print >> f, 'NX = {0}'.format(self.nx)
            print >> f, 'NY = {0}'.format(self.ny)
            print >> f, ''
            print >> f, 'u, v, y_r, y_i, noise_std_dev'
            print >> f, ''
            for i in xrange(self.m):
                print >> f, '{0}, {1}, {2:e}, {3:e}, {4:e}'.format(self.u[i],
                                                                   self.v[i],
                                                                   self.yreal[i],
                                                                   self.yimag[i],
                                                                   self.noise[i])
    
class SparseImagingResults(CTypesUtilMixIn):
    class MFISTAResult(ctypes.Structure):
        _fields_ = [('M', ctypes.c_int),
                    ('N', ctypes.c_int),
                    ('NX', ctypes.c_int),
                    ('NY', ctypes.c_int),
                    ('N_active', ctypes.c_int),
                    ('maxiter', ctypes.c_int),
                    ('ITER', ctypes.c_int),
                    ('nonneg', ctypes.c_int),
                    ('lambda_l1', ctypes.c_double),
                    ('lambda_tv', ctypes.c_double),
                    ('lambda_tsv', ctypes.c_double),
                    ('sq_error', ctypes.c_double),
                    ('mean_sq_error', ctypes.c_double),
                    ('l1cost', ctypes.c_double),
                    ('tvcost', ctypes.c_double),
                    ('tsvcost', ctypes.c_double),
                    ('looe_m', ctypes.c_double),
                    ('Hessian_positive', ctypes.c_double),
                    ('finalcost', ctypes.c_double),
                    ('comp_time', ctypes.c_double),
                    ('residual', ctypes.c_void_p)]
        
    def __init__(self, nx, ny, initialimage=None):
        self.nx = nx
        self.ny = ny
        nn = nx * ny
        self.xinit = numpy.empty(nn, dtype=numpy.double)
        if initialimage is None:
            # by default, initially all pixels are 1.0
            self.xinit[:] = 1.0
        else:
            # initial image is set by the user
            assert isinstance(initialimage, numpy.ndarray) or isinstance(initialimage, list)
            assert len(initialimage) == nn
            self.xinit[:] = initialimage
            
        self.xout = numpy.empty_like(self.xinit)
        self.mfista_result = self.MFISTAResult()
        
    @property
    def image(self):
        img = self.xout.reshape((self.nx,self.ny))
        return img

class SparseImagingExecutor(object):
    """
    ./mfista_imaging_fft fft_data.txt 1 0.0 0.01 5e10 x.out -nonneg
    """
    default_path = '/Users/nakazato/development/sparseimaging/20170812.mfista/'
    #libname = 'mfista_imaging_fft'
    libname = 'libmfista_fft.so'
    def __init__(self, lambda_L1, lambda_TV=0.0, lambda_TSV=0.0, 
                 cinit=5e10, nonnegative=True,
                 libpath=None):
        self.lambda_L1 = lambda_L1
        self.lambda_TV = lambda_TV
        self.lambda_TSV = lambda_TSV
        self.cinit = cinit
        self.nonnegative = nonnegative
        self.libpath = self.default_path if libpath is None else libpath
        
        nx = None
        ny = None
        self.outfile = 'x.out'

        # load library
        cdll = ctypes.cdll
        _mfista_name = os.path.join(self.libpath, self.libname)
        self._mfista = cdll.LoadLibrary(_mfista_name)
    
    def run(self, inputs, initialimage=None,
            maxiter=50000, eps=1.0e-5, cl_box=None):
        """
        Run MFISTA routine to get an image
        
        signature is 
        
          mfista_imaging_core_fft(int *u_idx, int *v_idx,
                 double *y_r, double *y_i, double *noise_stdev,
                 int M, int NX, int NY, int maxiter, double eps,
                 double lambda_l1, double lambda_tv, double lambda_tsv,
                 double cinit, double *xinit, double *xout,
                 int nonneg_flag, unsigned int fftw_plan_flag,
                 int box_flag, float *cl_box,
                 struct RESULT *mfista_result)
        """
        # input summary
        print 'lambda_l1 = {0}'.format(self.lambda_L1)
        print 'lambda_tv = {0}'.format(self.lambda_TV)
        print 'lambda_tsv = {0}'.format(self.lambda_TSV)
        print 'c = {0:g}'.format(self.cinit)
        print ''
        print 'number of u-v points: {0}'.format(inputs.m)
        print 'X-dim of image:       {0}'.format(inputs.nx)
        print 'Y-dim of image:       {0}'.format(inputs.ny)
        
        # inputs
        u_idx = ctypes.pointer(inputs.as_carray('u'))
        v_idx = ctypes.pointer(inputs.as_carray('v'))
        y_r = ctypes.pointer(inputs.as_carray('yreal'))
        y_i = ctypes.pointer(inputs.as_carray('yimag'))
        noise_stdev = ctypes.pointer(inputs.as_carray('noise'))
        M = ctypes.c_int(inputs.m)
        NX = ctypes.c_int(inputs.nx)
        NY = ctypes.c_int(inputs.ny)
        lambda_l1 = ctypes.c_double(self.lambda_L1)
        lambda_tv = ctypes.c_double(self.lambda_TV)
        lambda_tsv = ctypes.c_double(self.lambda_TSV)
        cinit = ctypes.c_double(self.cinit)
        nonneg_flag = ctypes.c_int(1 if self.nonnegative else 0)
        _maxiter = ctypes.c_int(maxiter)
        _eps = ctypes.c_double(eps)
        box_flag = 0 if cl_box is None else 1
        if box_flag == 1:
            cl_box = numpy.ctypeslib.as_ctypes(clean_box)
        else:
            cl_box = numpy.ctypeslib.as_ctypes(numpy.zeros(1, dtype=numpy.float32))
        _box_flag = ctypes.c_int(box_flag)
        fftw_plan_flag = ctypes.c_uint(65) # FFTW_ESTIMATE | FFTW_DESTROY_INPUT
        
        # outputs
        result = SparseImagingResults(inputs.nx, inputs.ny, initialimage=initialimage)
        xinit = ctypes.pointer(result.as_carray('xinit'))
        xout = ctypes.pointer(result.as_carray('xout'))
        mfista_result = ctypes.pointer(result.mfista_result)
        
        # run MFISTA
        self._mfista.mfista_imaging_core_fft(u_idx, v_idx, y_r, y_i, noise_stdev,
                                             M, NX, NY, _maxiter, _eps,
                                             lambda_l1, lambda_tv, lambda_tsv, 
                                             cinit, xinit, xout, nonneg_flag, fftw_plan_flag,
                                             _box_flag, cl_box, mfista_result)
        
        # show IO filenames
        self._show_io_info(inputs, initialimage)
        
        # show result
        self._show_result(result.mfista_result)
        
        return result
        
    def _show_io_info(self, inputs, initialimage=None):
        # show IO filenames
        print ''
        print ''
        print 'IO files of {0}.'.format(self.libname)
        print ''
        print ''
        print ' FFTW file:              {0}'.format(inputs.infile)
        if initialimage is None:
            print ' x was initialized with 1.0'
        else:
            print ' x was initialize by the user'
        #print ' x is saved to:          xout'
        print ''
        
        
    def _show_result(self, mfista_result):
        # show results
        print ''
        print ''
        print 'Output of {0}.'.format(self.libname)
        print ''
        print ''
        print ' Size of the problem:'
        print ''
        print ''
        print ' size of input vector:  {0}'.format(mfista_result.M)
        print ' size of output vector: {0}'.format(mfista_result.N)
        if mfista_result.NX != 0:
            print 'size of image:          {0} x {0}'.format(mfista_result.NX, 
                                                             mfista_result.NY)
        print ''
        print ''
        print ' Problem Setting:'
        print ''
        print ''
        if mfista_result.nonneg == 1:
            print ' x is a nonnegative vector.'
        elif mfista_result.nonneg == 0:
            print ' x is a real vector (takes 0, positive, and negative value).'
        print ''
        print ''
        if mfista_result.lambda_l1 != 0:
            print ' Lambda_l1: {0:e}'.format(mfista_result.lambda_l1)
        if mfista_result.lambda_tsv != 0:
            print ' Lambda_tsv: {0:e}'.format(mfista_result.lambda_tsv)
        if mfista_result.lambda_tv != 0:
            print ' Lambda_tv: {0:e}'.format(mfista_result.lambda_tv)
        print ' MAXITER: {0}'.format(mfista_result.maxiter)
        
        print ' Results:'
        print ''
        print ' # of iterations:       {0}'.format(mfista_result.ITER)
        print ' cost:                  {0:e}'.format(mfista_result.finalcost)
        print ' computation time[sec]: {0:e}'.format(mfista_result.comp_time)
        print ''
        print ' # of nonzero pixels:   {0}'.format(mfista_result.N_active)
        print ' Squared Error (SE):    {0:e}'.format(mfista_result.sq_error)
        print ' Mean SE:               {0:e}'.format(mfista_result.mean_sq_error)
        if mfista_result.lambda_l1 != 0:
            print ' L1 cost:               {0:e}'.format(mfista_result.l1cost)
        if mfista_result.lambda_tsv != 0:
            print ' TSV cost:              {0:e}'.format(mfista_result.tsvcost)
        if mfista_result.lambda_tv != 0:
            print ' TV cost:               {0:e}'.format(mfista_result.tvcost)
        print ''
        print ' LOOE:    Could not be computed because Hessian was not positive definite.'
                
                
    def _exec_line(self, f, varname):
        line = f.readline()
        exec(line.rstrip('\n'))
        val = locals()[varname]
        #print '{0} = {1}'.format(varname, val)
        return val
    
    def read_input(self, infile):
        """
        Read input text data for FFT based MFISTA imaging
        """
        with open(infile, 'r') as f:
            # read M
            M = self._exec_line(f, 'M')
            
            # read NX
            NX = self._exec_line(f, 'NX')
            
            # read NY
            NY = self._exec_line(f, 'NY')
            
            # skip headers
            f.readline()
            f.readline()
            f.readline()
            
            # read input data
            u = numpy.empty(M, dtype=numpy.int32)
            v = numpy.empty_like(u)
            yreal = numpy.empty(M, dtype=numpy.double)
            yimag = numpy.empty_like(yreal)
            noise = numpy.empty_like(yreal)
            for i in xrange(M):
                line = f.readline()
                values = line.split(',')
                u[i] = numpy.int32(values[0].strip())
                v[i] = numpy.int32(values[1].strip())
                yreal[i] = numpy.double(values[2].strip())
                yimag[i] = numpy.double(values[3].strip())
                noise[i] = numpy.double(values[4].strip())
                #print '{0} {1} {2} {3}'.format(u[i], v[i], yreal[i], yimag[i], noise[i])
                
            inputs = SparseImagingInputs(infile, M, NX, NY, u, v, yreal, yimag, noise)
            return inputs
            
    
    def get_result(self, outfile):
        n = self.nx * self.ny
        arraydata = numpy.fromfile(outfile, dtype=numpy.double)
        assert len(arraydata) == n
        
        img = arraydata.reshape((self.nx,self.ny))
        
        # flip along longitude axis
        img = numpy.fliplr(img)
        
        return img
    

# utility
def plot_inputs(inputs, interpolation='nearest', coverage=False):
    areal = numpy.zeros((inputs.nx, inputs.ny,), dtype=numpy.float)
    aimag = numpy.zeros_like(areal)
    for i in xrange(inputs.m):
        areal[inputs.u[i], inputs.v[i]] = inputs.yreal[i]
        aimag[inputs.u[i], inputs.v[i]] = inputs.yimag[i]
        
    if coverage:
        areal[areal.nonzero()] = 1.0
        aimag[areal.nonzero()] = 1.0
    
    import pylab as pl
    pl.figure('REAL')
    pl.clf()
    pl.imshow(areal, interpolation=interpolation)
    pl.colorbar()
    pl.figure('IMAG')
    pl.clf()
    pl.imshow(aimag, interpolation=interpolation)
    pl.colorbar()
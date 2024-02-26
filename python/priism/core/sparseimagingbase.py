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
from __future__ import print_function

import glob
import os
import numpy as np
import ctypes
import pylab as pl


class CTypesUtilMixIn(object):
    def as_carray(self, attr):
        array = getattr(self, attr)
        return np.ctypeslib.as_ctypes(array)


def get_actual_library_name(library_dir, library_name):
    library_prefix, library_suffix = library_name.split('.')
    libraries = glob.glob(f'{library_prefix}*.{library_suffix}', root_dir=library_dir)
    assert len(libraries) == 1
    return os.path.join(library_dir, libraries[0])


def exec_line(f, varname):
    line = f.readline()
    exec(line.rstrip('\n'))
    val = locals()[varname]
    #print '{0} = {1}'.format(varname, val)
    return val


def __shift_with(n, iarr, shift_term, inplace=True):
    if inplace:
        ret = iarr
    else:
        ret = np.zeros_like(iarr)
    ret = (iarr + shift_term) % n
    return ret


def shift_uvindex(n, iarr, inplace=True):
    """
    Assuming that input array index, iarr, is configured so that
    zero-frequency term comes to the center, shift_uvindex shifts
    iarr so that zero-frequency term comes to the first element.
    It corresponds to np.fft.ifftshift.

    if n is odd:
        (a,b,c,d,e,f,g) -> (d,e,f,g,a,b,c)
    elif n is even:
        (a,b,c,d,e,f)   -> (d,e,f,a,b,c)

    n --- number of pixels along the axis
    iarr --- input array index
    inplace --- if True, iarr is edited instead to prepare output array
    """
    shift_term = n // 2
    return __shift_with(n, iarr, shift_term, inplace)


def rshift_uvindex(n, iarr, inplace=True):
    """
    Assuming that input array index, iarr, is configured so that
    zero-frequency term comes to the first element, rshift_uvindex
    shifts iarr so that zero-frequency term comes to the center.
    It corresponds to np.fft.fftshift.

    if n is odd:
        (a,b,c,d,e,f,g) -> (e,f,g,a,b,c,d)
    elif n is even:
        (a,b,c,d,e,f)   -> (d,e,f,a,b,c)

    n --- number of pixels along the axis
    iarr --- input array index
    inplace --- if True, iarr is edited instead to prepare output array
    """
    shift_term = n // 2 + (n % 2)
    return __shift_with(n, iarr, shift_term, inplace)


class SparseImagingInputs(CTypesUtilMixIn):
    """
    Container for sparseimaging inputs
    """
    @classmethod
    def from_file(cls, filename):
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
            u = np.empty(M, dtype=np.int32)
            v = np.empty_like(u)
            yreal = np.empty(M, dtype=np.double)
            yimag = np.empty_like(yreal)
            noise = np.empty_like(yreal)
            for i in range(M):
                line = f.readline()
                values = line.split(',')
                u[i] = np.int32(values[0].strip())
                v[i] = np.int32(values[1].strip())
                yreal[i] = np.double(values[2].strip())
                yimag[i] = np.double(values[3].strip())
                noise[i] = np.double(values[4].strip())
                #print '{0} {1} {2} {3}'.format(u[i], v[i], yreal[i], yimag[i], noise[i])

            inputs = cls(filename, M, NX, NY, u, v, yreal, yimag, noise)
            return inputs

    @classmethod
    def from_gridder_result(cls, gridder_result, imageparam):
        """
        Convert GridderResult object into SparseImagingInputs object.
        uv-coordinate value is flipped for FFTW.
        """
        # infile is nominal value
        infile = 'gridder_result'

        # m is the number of nonzero pixels
        nonzeros = np.nonzero(gridder_result.wreal)
        m = len(nonzeros[0])

        # np.nonzero returns index as np.int64
        # however, libmfista_fft requires index in 32bit integer
        # value check is performed here
        iint32 = np.iinfo(np.int32)
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
        unflipped_v = np.asarray(nonzeros[0], dtype=np.int32)
        unflipped_u = np.asarray(nonzeros[1], dtype=np.int32)
        u = shift_uvindex(nu, unflipped_u)
        v = shift_uvindex(nv, unflipped_v)

        # yreal, yimag are nonzero gridded visibility
        yreal = gridder_result.real[nonzeros]
        yimag = gridder_result.imag[nonzeros]

        # 20171102 suggestion by Ikeda-san
        # change sign according to pixel coordinate
        for i in range(len(yreal)):
            j = nonzeros[0][i]
            k = nonzeros[1][i]
            factor = (-1)**(j + k)
            yreal[i] *= factor
            yimag[i] *= factor

        # noise is formed as 1 / sqrt(weight)
        noise = gridder_result.wreal[nonzeros]
        noise = 1.0 / np.sqrt(noise)

        return cls(infile, m, nx, ny, u, v, yreal, yimag, noise)

    @classmethod
    def convert_uv(cls, imageparam, u, v):
        raise NotImplementedError('convert_uv must be implemented in subclasses!')

    @classmethod
    def convert_vis(cls, u, v, yreal, yimag):
        raise NotImplementedError('convert_vis must be implemented in subclasses!')

    @classmethod
    def from_visibility_working_set(cls, visibility, imageparam):
        """
        Convert VisibilityWorkingSet object into SparseImagingInputs object.
        uv-coordinate value is flipped for FFTW.
        """
        # infile is nominal value
        infile = 'visibility_working_set'

        # m is the number of visibility data
        m = len(visibility.u)

        # nx, ny
        nx = imageparam.imsize[0]
        ny = imageparam.imsize[1]

        u, v = cls.convert_uv(imageparam, visibility.u, visibility.v)

        yreal, yimag = cls.convert_vis(visibility.u, visibility.v, visibility.rdata, visibility.idata)

        # noise is formed as 1 / sqrt(weight)
        noise = visibility.weight.copy()
        noise = 1.0 / np.sqrt(noise)

        return cls(infile, m, nx, ny, u, v, yreal, yimag, noise)

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

    @property
    def header(self):
        return 'u, v, y_r, y_i, noise_std_dev'

    def export(self, filename):
        with open(filename, 'w') as f:
            print('M = {0}'.format(self.m), file=f)
            print('NX = {0}'.format(self.nx), file=f)
            print('NY = {0}'.format(self.ny), file=f)
            print('', file=f)
            print(self.header, file=f)
            print('', file=f)
            for i in range(self.m):
                print('{0}, {1}, {2:e}, {3:e}, {4:e}'.format(self.u[i],
                                                             self.v[i],
                                                             self.yreal[i],
                                                             self.yimag[i],
                                                             self.noise[i]), file=f)


class SparseImagingResults(CTypesUtilMixIn):
    ResultClass = None

    def __init__(self, nx, ny, initialimage=None):
        self.nx = nx
        self.ny = ny
        nn = nx * ny
        self.xinit = np.empty(nn, dtype=np.double)
        if initialimage is None:
            # by default, initially all pixels are 1.0
            self.xinit[:] = 0.0
        else:
            # initial image is set by the user
            assert isinstance(initialimage, np.ndarray) or isinstance(initialimage, list)
            assert len(initialimage) == nn
            self.xinit[:] = initialimage

        self.xout = np.empty_like(self.xinit)
        self.mfista_result = self.ResultClass()

    @property
    def image(self):
        img = self.xout.reshape((self.nx, self.ny))
        return img


class SparseImagingExecutor(object):
    """
    ./mfista_imaging_fft fft_data.txt 1 0.0 0.01 5e10 x.out -nonneg
    """
    Inputs = SparseImagingInputs
    #default_path = '/Users/nakazato/development/sparseimaging/20170812.mfista/'
    default_path = os.path.dirname(__file__)
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
        self.libpath = self.default_path  # if libpath is None else libpath

        nx = None
        ny = None
        self.outfile = 'x.out'

        # load library
        cdll = ctypes.cdll
        _mfista_name = get_actual_library_name(self.libpath, self.libname)
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
        print('lambda_l1 = {0}'.format(self.lambda_L1))
        print('lambda_tv = {0}'.format(self.lambda_TV))
        print('lambda_tsv = {0}'.format(self.lambda_TSV))
        print('c = {0:g}'.format(self.cinit))
        print('')
        print('number of u-v points: {0}'.format(inputs.m))
        print('X-dim of image:       {0}'.format(inputs.nx))
        print('Y-dim of image:       {0}'.format(inputs.ny))

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
            cl_box = np.ctypeslib.as_ctypes(cl_box)
        else:
            cl_box = np.ctypeslib.as_ctypes(np.zeros(1, dtype=np.float32))
        _box_flag = ctypes.c_int(box_flag)
        fftw_plan_flag = ctypes.c_uint(65)  # FFTW_ESTIMATE | FFTW_DESTROY_INPUT

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
                                             _box_flag, cl_box,
                                             mfista_result)

        # show IO filenames
        self._show_io_info(inputs, initialimage)

        # show result
        self._show_result(result.mfista_result)

        return result

    def _show_io_info(self, inputs, initialimage=None):
        # show IO filenames
        print('')
        print('')
        print('IO files of {0}.'.format(self.libname))
        print('')
        print('')
        print(' FFTW file:              {0}'.format(inputs.infile))
        if initialimage is None:
            print(' x was initialized with 0.0')
        else:
            print(' x was initialize by the user')
        #print ' x is saved to:          xout'
        print('')

    def _show_result(self, mfista_result):
        # show results
        print('')
        print('')
        print('Output of {0}.'.format(self.libname))
        print('')
        print('')
        print(' Size of the problem:')
        print('')
        print('')
        print(' size of input vector:  {0}'.format(mfista_result.M))
        print(' size of output vector: {0}'.format(mfista_result.N))
        if mfista_result.NX != 0:
            print('size of image:          {0} x {1}'.format(mfista_result.NX,
                                                             mfista_result.NY))
        print('')
        print('')
        print(' Problem Setting:')
        print('')
        print('')
        if mfista_result.nonneg == 1:
            print(' x is a nonnegative vector.')
        elif mfista_result.nonneg == 0:
            print(' x is a real vector (takes 0, positive, and negative value).')
        print('')
        print('')
        if mfista_result.lambda_l1 != 0:
            print(' Lambda_l1: {0:e}'.format(mfista_result.lambda_l1))
        if mfista_result.lambda_tsv != 0:
            print(' Lambda_tsv: {0:e}'.format(mfista_result.lambda_tsv))
        if mfista_result.lambda_tv != 0:
            print(' Lambda_tv: {0:e}'.format(mfista_result.lambda_tv))
        print(' MAXITER: {0}'.format(mfista_result.maxiter))

        print(' Results:')
        print('')
        print(' # of iterations:       {0}'.format(mfista_result.ITER))
        print(' cost:                  {0:e}'.format(mfista_result.finalcost))
        print(' computation time[sec]: {0:e}'.format(mfista_result.comp_time))
        print('')
        print(' # of nonzero pixels:   {0}'.format(mfista_result.N_active))
        print(' Squared Error (SE):    {0:e}'.format(mfista_result.sq_error))
        print(' Mean SE:               {0:e}'.format(mfista_result.mean_sq_error))
        if mfista_result.lambda_l1 != 0:
            print(' L1 cost:               {0:e}'.format(mfista_result.l1cost))
        if mfista_result.lambda_tsv != 0:
            print(' TSV cost:              {0:e}'.format(mfista_result.tsvcost))
        if mfista_result.lambda_tv != 0:
            print(' TV cost:               {0:e}'.format(mfista_result.tvcost))
        print('')
        print(' LOOE:    Could not be computed because Hessian was not positive definite.')

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
            u = np.empty(M, dtype=np.int32)
            v = np.empty_like(u)
            yreal = np.empty(M, dtype=np.double)
            yimag = np.empty_like(yreal)
            noise = np.empty_like(yreal)
            for i in range(M):
                line = f.readline()
                values = line.split(',')
                u[i] = np.int32(values[0].strip())
                v[i] = np.int32(values[1].strip())
                yreal[i] = np.double(values[2].strip())
                yimag[i] = np.double(values[3].strip())
                noise[i] = np.double(values[4].strip())
                #print '{0} {1} {2} {3}'.format(u[i], v[i], yreal[i], yimag[i], noise[i])

            inputs = SparseImagingInputs(infile, M, NX, NY, u, v, yreal, yimag, noise)
            return inputs

    def get_result(self, outfile):
        n = self.nx * self.ny
        arraydata = np.fromfile(outfile, dtype=np.double)
        assert len(arraydata) == n

        img = arraydata.reshape((self.nx, self.ny))

        # flip along longitude axis
        img = np.fliplr(img)

        return img


# utility
def plot_inputs(inputs, interpolation='nearest', coverage=False):
    areal = np.zeros((inputs.nx, inputs.ny,), dtype=np.float)
    aimag = np.zeros_like(areal)
    for i in range(inputs.m):
        areal[inputs.u[i], inputs.v[i]] = inputs.yreal[i]
        aimag[inputs.u[i], inputs.v[i]] = inputs.yimag[i]

    if coverage:
        areal[areal.nonzero()] = 1.0
        aimag[areal.nonzero()] = 1.0

    pl.figure('REAL')
    pl.clf()
    pl.imshow(areal, interpolation=interpolation)
    pl.colorbar()
    pl.figure('IMAG')
    pl.clf()
    pl.imshow(aimag, interpolation=interpolation)
    pl.colorbar()

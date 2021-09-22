# Copyright (C) 2019
# National Astronomical Observatory of Japan
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

import os
import numpy as np
import cmath
import ctypes

import priism.external.casa.casatools as casatools
from . import sparseimagingbase


class SparseImagingInputsClosure(sparseimagingbase.SparseImagingInputs):
    @classmethod
    def from_visibility_working_set(cls, visibility, imageparam):
        """
        Convert VisibilityWorkingSet object into SparseImagingInputs object.
        uv-coordinate value is flipped for FFTW.
        """
        base_inputs = sparseimagingbase.SparseImagingInputs.from_visibility_working_set(
            visibility,
            imageparam
        )
        t = visibility.t
        a1 = visibility.a1
        a2 = visibility.a2
        chan = visibility.chan
        weight = visibility.weight
        return cls(base_inputs.infile, base_inputs.m, base_inputs.nx, base_inputs.ny,
                   t, a1, a2, chan,
                   base_inputs.u, base_inputs.v, visibility.w,
                   base_inputs.yreal, base_inputs.yimag,
                   base_inputs.noise, weight)

    @property
    def header(self):
        return 'mjd,year,month,day,hour,min,sec,freq,stokesid,bandid,ifid,u,v,w,st1,st2,amp,phase,weight,sigma'

    def __init__(self, infile, M, NX, NY, t, a1, a2, ch, u, v, w, yreal, yimag, noise, weight):
        super(SparseImagingInputsClosure, self).__init__(infile, M, NX, NY, u, v, yreal, yimag, noise)
        self.t = t
        self.a1 = a1
        self.a2 = a2
        self.ch = ch
        self.w = w
        self.weight = weight

    def export(self, filename):
        """Export data to file.

        Requirement from Ikeda-san
        mjd,year,month,day,hour,min,sec,freq,stokesid,bandid,ifid,u,v,w,st1,st2,amp,phase,weight,sigma,weight

        Args:
            filename (str): output file name
        """
        qa = casatools.CasaToolGenerator.CreateQuantity()
        with open(filename, 'w') as f:
            # print('M = {0}'.format(self.m), file=f)
            # print('NX = {0}'.format(self.nx), file=f)
            # print('NY = {0}'.format(self.ny), file=f)
            # print('', file=f)
            print(f'#{self.header}', file=f)
            for i in range(self.m):
                qtime = qa.quantity(self.t[i], 's')
                time_dict = qa.splitdate(qtime)
                visibility = complex(self.yreal[i], self.yimag[i])
                amp = abs(visibility)
                phase = cmath.phase(visibility)
                print('{time}, {year:d}, {month:d}, {day:d}, {hour:d}, {minute:d}, {sec}, {freq}, {stokes:d}, {band:d}, {ifid:d}, {u}, {v}, {w}, {st1}, {st2}, {amp:e}, {phase:e}, {weight:e}, {sigma:e}'.format(
                    time=time_dict['mjd'],
                    year=int(time_dict['year']),
                    month=int(time_dict['month']),
                    day=int(time_dict['monthday']),
                    hour=int(time_dict['hour']),
                    minute=int(time_dict['min']),
                    sec=time_dict['sec'] + time_dict['usec'] * 1e-6,
                    freq=self.ch[i],  # temporary: will be replaced with channel frequency
                    stokes=0,  # TEMPORARY HARD-CODED: Stokes I
                    band=6,  # TEMPORARY HARD-CODED: Band 6
                    ifid=0,  # TEMPORARY HARD-CODED: spw 0
                    u=self.u[i],
                    v=self.v[i],
                    w=self.w[i],
                    st1=self.a1[i],
                    st2=self.a2[i],
                    amp=amp,
                    phase=phase,
                    weight=self.weight[i],
                    sigma=self.noise[i]
                ), file=f)


class MFISTAResultClosure(ctypes.Structure):
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
                ('finalcost', ctypes.c_double),
                ('comp_time', ctypes.c_double),
                ('residual', ctypes.c_void_p),
                ('Lip_const', ctypes.c_double)]


class SparseImagingResultsClosure(sparseimagingbase.SparseImagingResults):
    ResultClass = MFISTAResultClosure


class SparseImagingExecutor(object):
    """
    """
    Inputs = SparseImagingInputsClosure
    #default_path = '/Users/nakazato/development/sparseimaging/20170812.mfista/'
    default_path = os.path.dirname(__file__)
    #libname = 'mfista_imaging_fft'
    libname = 'libmfista_nufft.so'

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
        _mfista_name = os.path.join(self.libpath, self.libname)
        self._mfista = cdll.LoadLibrary(_mfista_name)

    def run(self, inputs, initialimage=None,
            maxiter=50000, eps=1.0e-5, cl_box=None):
        """
        Run MFISTA routine to get an image

        signature is

        void mfista_imaging_core_nufft(double *u_dx, double *v_dy,
                   double *vis_r, double *vis_i, double *vis_std,
                   int M, int Nx, int Ny, int maxiter, double eps,
                   double lambda_l1, double lambda_tv, double lambda_tsv,
                   double cinit, double *xinit, double *xout,
                   int nonneg_flag, int box_flag, float *cl_box,
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
        assert inputs.yreal.dtype == np.float64, 'yreal.dtype = {}'.format(inputs.yreal.dtype)
        assert inputs.yimag.dtype == np.float64, 'yimag.dtype = {}'.format(inputs.yimag.dtype)
        assert inputs.noise.dtype == np.float64, 'noise.dtype = {}'.format(inputs.noise.dtype)
        y_r = ctypes.pointer(inputs.as_carray('yreal'))
        y_i = ctypes.pointer(inputs.as_carray('yimag'))
        noise_stdev = ctypes.pointer(inputs.as_carray('noise'))
        M = ctypes.c_int(inputs.m)
        NX = ctypes.c_int(inputs.nx)
        NY = ctypes.c_int(inputs.ny)
        _maxiter = ctypes.c_int(maxiter)
        _eps = ctypes.c_double(eps)
        lambda_l1 = ctypes.c_double(self.lambda_L1)
        lambda_tv = ctypes.c_double(self.lambda_TV)
        lambda_tsv = ctypes.c_double(self.lambda_TSV)
        cinit = ctypes.c_double(self.cinit)
        nonneg_flag = ctypes.c_int(1 if self.nonnegative else 0)
        box_flag = 0 if cl_box is None else 1
        if box_flag == 1:
            cl_box = np.ctypeslib.as_ctypes(cl_box)
        else:
            cl_box = np.ctypeslib.as_ctypes(np.zeros(1, dtype=np.float32))
        _box_flag = ctypes.c_int(box_flag)

        # outputs
        result = SparseImagingResultsClosure(inputs.nx, inputs.ny, initialimage=initialimage)
        xinit = ctypes.pointer(result.as_carray('xinit'))
        xout = ctypes.pointer(result.as_carray('xout'))
        mfista_result = ctypes.pointer(result.mfista_result)

        # run MFISTA
        self._mfista.mfista_imaging_core_nufft(u_idx, v_idx, y_r, y_i, noise_stdev,
                                               M, NX, NY, _maxiter, _eps,
                                               lambda_l1, lambda_tv, lambda_tsv,
                                               cinit, xinit, xout, nonneg_flag,
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
            print(' x was initialized with 1.0')
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

            inputs = self.Inputs(infile, M, NX, NY, u, v, yreal, yimag, noise)
            return inputs

    def get_result(self, outfile):
        n = self.nx * self.ny
        arraydata = np.fromfile(outfile, dtype=np.double)
        assert len(arraydata) == n

        img = arraydata.reshape((self.nx, self.ny))

        # flip along longitude axis
        img = np.fliplr(img)

        return img

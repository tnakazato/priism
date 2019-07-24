from __future__ import absolute_import

import numpy
import libsakurapy


def sakura_typemap(dtype):
    if dtype in (numpy.float64, numpy.double,):
        return libsakurapy.TYPE_DOUBLE
    elif dtype in (numpy.float32, float,):
        return libsakurapy.TYPE_FLOAT
    elif dtype in (numpy.bool, bool,):
        return libsakurapy.TYPE_BOOL
    elif dtype in (numpy.int32, numpy.int, int,):
        return libsakurapy.TYPE_INT32
    elif dtype in (numpy.uint32, numpy.uint,):
        return libsakurapy.TYPE_UINT32
    elif dtype in (numpy.int8,):
        return libsakurapy.TYPE_INT8
    elif dtype in (numpy.uint8,):
        return libsakurapy.TYPE_UINT8
    else:
        raise NotImplementedError('data type {0} is not supported by sakura allocation function'.format(dtype))


def empty_aligned(shape, dtype=float, order='C', alignment=None):
    """
    Equivalent to numpy.empty. Array data is assured to be aligned
    to byte boundary specified by alignment. Default alignment is
    the one required by sakura.

    shape --- array shape
    dtype --- data type
    order --- ignored
    alignment --- data alignment requirement. Default (None) indicates
                  the one for sakura. currently only None is allowed
    """
    sakura_type = sakura_typemap(dtype)
    return libsakurapy.new_uninitialized_aligned_ndarray(sakura_type, shape)


def empty_like_aligned(array, alignment=None):
    order = 'F' if numpy.isfortran(array) else 'C'
    return empty_aligned(shape=array.shape, dtype=array.dtype,
                         order=order, alignment=alignment)

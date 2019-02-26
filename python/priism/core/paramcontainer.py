from __future__ import absolute_import

class ParamContainer(object):
    def InitContainer(self, kwargs):
        ignores = ('self',)
        for k,v in kwargs.items():
            if k not in ignores:
                setattr(self, k, v)

    @classmethod
    def CreateContainer(container_cls, **kwargs):
        ignores = ('self',)
        kw = kwargs.copy()
        for key in ignores:
            if key in kw:
                kw.pop(key)
        return container_cls(**kw)


class MfistaParamContainer(ParamContainer):
    def __init__(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None):
        """
        Constructor
        """
        self.InitContainer(locals())

class SimpleImageParamContainer(ParamContainer):
    """
    This is primitive image parameter container that 
    specifies three-dimensional cube (imsize for spatial axes 
    and nchan for spectral axis).
    """
    def __init__(self, imsize=100, nchan=-1):
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
    
        
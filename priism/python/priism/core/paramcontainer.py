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

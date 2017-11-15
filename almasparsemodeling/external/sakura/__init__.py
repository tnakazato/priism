from __future__ import absolute_import

import numpy

from .reductionhelper import paraMap
import libsakurapy

from ..casa import casa_atexit

# libsakurapy is initialized when the module is imported
# and is cleaned up at exit
try:
    if _SAKURA_INITIALIZED_:
        pass
except NameError:
    _SAKURA_INITIALIZED_ = True
    print 'LOG: initialize sakura...'
    print libsakurapy
    libsakurapy.initialize()
    def sakura_cleanup():
        print 'LOG: clean up sakura...'
        libsakurapy.clean_up()
    casa_atexit.register(sakura_cleanup)

from .allocator import empty_aligned
from .allocator import empty_like_aligned

from .core import grid
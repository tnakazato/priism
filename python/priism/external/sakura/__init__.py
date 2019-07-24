from __future__ import absolute_import

import atexit

from .reductionhelper import paraMap
import libsakurapy

# libsakurapy is initialized when the module is imported
# and is cleaned up at exit
try:
    if _SAKURA_INITIALIZED_:
        pass
except NameError:
    _SAKURA_INITIALIZED_ = True
    print('LOG: initialize sakura...')
    print(libsakurapy)
    libsakurapy.initialize()

    def sakura_cleanup():
        print('LOG: clean up sakura...')
        libsakurapy.clean_up()
    atexit.register(sakura_cleanup)

from .allocator import empty_aligned
from .allocator import empty_like_aligned

from .core import grid

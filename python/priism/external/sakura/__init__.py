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

import atexit

from .reductionhelper import paraMap
from . import libsakurapy

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

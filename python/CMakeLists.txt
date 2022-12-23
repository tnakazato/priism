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
cmake_minimum_required(VERSION 2.8.12)

message(STATUS "Entering priism subdirectory...")

install(FILES priism/__init__.py
        DESTINATION priism)

install(FILES
        priism/core/__init__.py
        priism/core/cv.py
        priism/core/datacontainer.py
        priism/core/imager.py
        priism/core/mfista.py
        priism/core/paramcontainer.py
        priism/core/sparseimagingbase.py
        priism/core/sparseimagingfft.py
        priism/core/sparseimagingnufft.py
        priism/core/util.py
        priism/core/version.py
        DESTINATION priism/core)

install(FILES
        priism/alma/__init__.py
        priism/alma/gridder.py
        priism/alma/imager.py
        priism/alma/imagewriter.py
        priism/alma/paramcontainer.py
        priism/alma/util.py
        priism/alma/visconverter.py
        priism/alma/visreader.py
        DESTINATION priism/alma)

install(FILES priism/external/__init__.py
        DESTINATION priism/external)

install(FILES
        priism/external/casa/__init__.py
        priism/external/casa/casa_atexit.py
        priism/external/casa/casatools.py
        DESTINATION priism/external/casa)

install(FILES
        priism/external/sakura/__init__.py
        priism/external/sakura/allocator.py
        priism/external/sakura/core.py
        priism/external/sakura/reductionhelper.py
        DESTINATION priism/external/sakura)

install(FILES
        priism/test/__init__.py
        #priism/test/demo.py
        priism/test/cvrun_fft.py
        priism/test/cvrun_nufft.py
        priism/test/cvrun_core.py
        priism/test/runtest.py
        priism/test/test_gridder.py
        priism/test/test_imagewriter.py
        priism/test/test_util.py
        priism/test/test_visconverter.py
        priism/test/test_visreader.py
        priism/test/utils.py
        DESTINATION priism/test)
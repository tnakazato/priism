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

import contextlib

try:
    # CASA 5
    import casac
    _casac = casac.casac()
except Exception:
    # CASA 6
    import casatools
    _casac = casatools

# logger
try:
    # CASA 5
    from taskinit import casalog
except Exception:
    # CASA 6
    from casatasks import casalog


class CasaToolGenerator(object):
    @staticmethod
    def _Create(name):
        tool = getattr(_casac, name)()
        return tool

    @staticmethod
    def CreateTable():
        return CasaToolGenerator._Create('table')

    @staticmethod
    def CreateMS():
        return CasaToolGenerator._Create('ms')

    @staticmethod
    def CreateMSMetaData():
        return CasaToolGenerator._Create('msmetadata')

    @staticmethod
    def CreateImageAnalysis():
        return CasaToolGenerator._Create('image')

    @staticmethod
    def CreateCoordSys():
        return CasaToolGenerator._Create('coordsys')

    @staticmethod
    def CreateMeasure():
        return CasaToolGenerator._Create('measures')

    @staticmethod
    def CreateQuantity():
        return CasaToolGenerator._Create('quanta')


@contextlib.contextmanager
def OpenTableForRead(vis):
    tb = CasaToolGenerator.CreateTable()
    tb.open(vis, nomodify=True)
    try:
        yield tb
    finally:
        tb.close()


@contextlib.contextmanager
def OpenTableForReadWrite(vis):
    tb = CasaToolGenerator.CreateTable()
    tb.open(vis, nomodify=False)
    try:
        yield tb
    finally:
        tb.close()


@contextlib.contextmanager
def SelectTableForRead(vis, taql):
    tb = CasaToolGenerator.CreateTable()
    tb.open(vis, nomodify=False)
    try:
        tsel = tb.query(taql)
        try:
            yield tsel
        finally:
            tsel.close()
    finally:
        tb.close()


@contextlib.contextmanager
def OpenMS(vis):
    ms = CasaToolGenerator.CreateMS()
    ms.open(vis)
    try:
        yield ms
    finally:
        ms.close()


@contextlib.contextmanager
def OpenMSMetaData(vis):
    msmd = CasaToolGenerator.CreateMSMetaData()
    msmd.open(vis)
    try:
        yield msmd
    finally:
        msmd.close()


@contextlib.contextmanager
def OpenImage(imagename):
    ia = CasaToolGenerator.CreateImageAnalysis()
    ia.open(imagename)
    try:
        yield ia
    finally:
        ia.close()

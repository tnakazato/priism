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

import contextlib

from taskinit import gentools, qa


class CasaToolGenerator(object):
    @staticmethod
    def _Create(name):
        (tool,) = gentools([name])
        return tool

    @staticmethod
    def CreateTable():
        return CasaToolGenerator._Create('tb')

    @staticmethod
    def CreateImageAnalysis():
        return CasaToolGenerator._Create('ia')

    @staticmethod
    def CreateCoordSys():
        return CasaToolGenerator._Create('cs')

    @staticmethod
    def CreateMeasure():
        return CasaToolGenerator._Create('me')

    @staticmethod
    def CreateQuantity():
        return qa


@contextlib.contextmanager
def OpenTableForRead(vis):
    (tb,) = gentools(['tb'])
    tb.open(vis, nomodify=True)
    try:
        yield tb
    finally:
        tb.close()


@contextlib.contextmanager
def OpenTableForReadWrite(vis):
    (tb,) = gentools(['tb'])
    tb.open(vis, nomodify=False)
    try:
        yield tb
    finally:
        tb.close()


@contextlib.contextmanager
def SelectTableForRead(vis, taql):
    (tb,) = gentools(['tb'])
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
    (ms,) = gentools(['ms'])
    ms.open(vis)
    try:
        yield ms
    finally:
        ms.close()


@contextlib.contextmanager
def OpenMSMetaData(vis):
    (msmd,) = gentools(['msmd'])
    msmd.open(vis)
    try:
        yield msmd
    finally:
        msmd.close()


@contextlib.contextmanager
def OpenImage(imagename):
    (ia,) = gentools(['ia'])
    ia.open(imagename)
    try:
        yield ia
    finally:
        ia.close()

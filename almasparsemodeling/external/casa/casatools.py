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
    
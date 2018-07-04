from __future__ import absolute_import


# casa tool generators
from .casatools import CasaToolGenerator
CreateCasaTable = CasaToolGenerator.CreateTable
CreateCasaImageAnalysis = CasaToolGenerator.CreateImageAnalysis
CreateCasaCoordSys = CasaToolGenerator.CreateCoordSys
CreateCasaMeasure = CasaToolGenerator.CreateMeasure
CreateCasaQuantity = CasaToolGenerator.CreateQuantity

# casa utility
from .casatools import OpenTableForRead
from .casatools import OpenTableForReadWrite
from .casatools import SelectTableForRead
from .casatools import OpenMS
from .casatools import OpenMSMetaData
from .casatools import OpenImage


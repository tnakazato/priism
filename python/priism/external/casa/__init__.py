from __future__ import absolute_import

import collections


# casa version
def _get_casa_version():
    try:
        import inspect
        version_string = 'NONE'
        frames = inspect.getouterframes(inspect.currentframe())
        for f in frames:
            if 'casa' in f[0].f_globals:
                casadict = f[0].f_globals['casa']
                if 'version' in casadict:
                    version_string = casadict['version']
                    break
                elif 'build' in casadict:
                    buildinfo = casadict['build']
                    if 'version' in buildinfo:
                        version_string = buildinfo['version']
                        break
    except Exception:
        version_string = 'NONE'
    #print(version_string)

    # version string format: MAJOR.MINOR.PATCH-BUILD
    s = version_string.split('-')
    if len(s) != 2:
        build = 0
    else:
        build = int(s[1])
    t = s[0].split('.')
    if len(t) != 3:
        major = -1
        minor = 0
        patch = 0
    else:
        major = int(t[0])
        minor = int(t[1])
        patch = int(t[2])
    CASAVersion = collections.namedtuple('CASAVersion', ['major', 'minor', 'patch', 'build'])
    return CASAVersion(major=major, minor=minor, patch=patch, build=build)


casa_version = _get_casa_version()


# Fail if CASA is not available, warn if CASA is too old
if casa_version.major < 0:
    raise ImportError('CASA tools/tasks is not available. prism.alma will not work. Please use prism or prism.core instead.')
elif casa_version.major < 5:
    print('WARNING: CASA version should be 5.0.0 or higher. prism.alma will not work. Please use prism or prism.core instead.')

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

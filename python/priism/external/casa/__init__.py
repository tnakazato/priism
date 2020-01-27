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

import collections
import functools


# casa version
def _get_casa_version():
    try:
        # first, try accessing casa dictionary
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
        raise Exception('no appropriate frame found')
    except Exception:
        try:
            # then, try using casac module
            import casac
            _casac = casac.casac()
            utils = _casac.utils()
            version_string = utils.version_info().strip()
            #print(version_string)
        except Exception:
            try:
                # this could be CASA 6 environment
                import casatools
                version_string = casatools.ctsys.version_string()
            except Exception:
                # no casa found...
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
from .casatools import casalog


def adjust_casalog_level(level='INFO'):
    def f(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            casalog.filter(level=level)
            try:
                ret = func(*args, **kwargs)
                return ret
            finally:
                casalog.filter(level='INFO')

        return wrapper

    return f

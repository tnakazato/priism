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

from . import paramcontainer
from . import visconverter
import priism.external.casa as casa


class VisibilityReader(object):
    def __method_name(self, basename):
        if casa.casa_version.major == 5:
            if casa.casa_version.minor >= 3:
                return basename
            elif casa.casa_version.minor >= 0:
                return '{}2'.format(basename)
        elif casa.casa_version.major >= 6:
            return basename
        raise RuntimeError('Unsupported CASA version {}.{}.{}-{}'.format(casa.casa_version.major,
                                                                         casa.casa_version.minor,
                                                                         casa.casa_version.major,
                                                                         casa.casa_version.build))

    @property
    def iterinit(self):
        return self.__method_name('iterinit')

    @property
    def iterorigin(self):
        return self.__method_name('iterorigin')

    @property
    def getdata(self):
        return self.__method_name('getdata')

    @property
    def iternext(self):
        return self.__method_name('iternext')

    """
    VisibilityReader is responsible for scanning and reading MS
    according to user-specified data selection (MS name,
    fields, spws, etc.).

    It has a generator method "readvis" to provide an
    access to all selected data.
    """
    def __init__(self, visparam):
        """
        Constructor

        visparam -- visibility data selection parameter
                    as an instance of VisParamContainer
        """
        self.visparam = visparam
        self.sel = None

        assert isinstance(self.visparam, paramcontainer.VisParamContainer)

    def readvis(self, items=visconverter.VisibilityConverter.required_columns,
                columns=[], interval=0.0, nrow=0, adddefault=True):
        """
        read visibility data accoding to the data selection
        provided to constructor.

        columns -- iteration axis
        interval -- time interval to group together
        nrow -- number of row for returned data chunk
        adddefault -- add default sort columns to iteration axis
        """
        vis = self.visparam.vis
        msselect = self.visparam.as_msselection()

        with casa.OpenMS(vis) as ms:
            ms.msselect(msselect, onlyparse=False)

            # method names depend on CASA version
            ms_iterinit = getattr(ms, self.iterinit)
            ms_iterorigin = getattr(ms, self.iterorigin)
            ms_getdata = getattr(ms, self.getdata)
            ms_iternext = getattr(ms, self.iternext)

            # not using new iterator as it doesn't support
            # reading meta information such as UVW...
            # iterate through MS using VI/VB2 framework
            ms_iterinit(columns, interval, nrow, adddefault)
            more_chunks = ms_iterorigin()
            chunk_id = 0
            while (more_chunks):
                rec = ms_getdata(items)
                rec['chunk_id'] = chunk_id
                #print('LOG: read visibility chunk #{0}'.format(chunk_id))
                yield rec
                more_chunks = ms_iternext()
                chunk_id += 1

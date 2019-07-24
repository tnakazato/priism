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

import unittest
import nose
import sys

from . import test_imagewriter
from . import test_visreader
from . import test_visconverter
from . import test_gridder
from . import test_util


def run_test(verbosity=2, tests=None):
    test_suite = unittest.TestSuite()
    modules = (test_imagewriter,
               test_visreader,
               test_visconverter,
               test_gridder,
               test_util)
    if tests is None:
        for m in modules:
            test_suite.addTests(m.suite())
    else:
        for m in modules:
            suite = m.suite()
            for case in suite:
                if m.__name__.split('.')[-1] in tests or \
                    case.__class__.__name__ in tests or \
                    case._testMethodName in tests:
                    test_suite.addTest(case)

    argv = [sys.argv[0],
            '--verbosity={0}'.format(verbosity),
            '--nocapture']

    nose.run(argv=argv, suite=test_suite)

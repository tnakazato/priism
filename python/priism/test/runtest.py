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

from __future__ import absolute_import

import unittest
import numpy


class TestBase(unittest.TestCase):
    def assertMaxDiffLess(self, ref, val, eps, msg=''):
        diff = numpy.abs((val - ref) / ref)
        if hasattr(diff, '__iter__'):
            diffmax = diff.max()
        else:
            diffmax = diff
        self.assertLess(diffmax, eps, msg=msg)

    def assertIndexListEqual(self, val, ref):
        self.assertEqual(len(val), len(ref))
        for (a, b) in zip(val, ref):
            self.assertEqual(len(a), len(b))
            self.assertTrue(numpy.all(a == b))


def generate_suite(test_cls, test_items):
    test_suite = unittest.TestSuite()
    for item in test_items:
        test_suite.addTest(test_cls(item))
    return test_suite

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

import unittest
import numpy as np


class TestBase(unittest.TestCase):
    def assertMaxDiffLess(self, ref, val, eps, msg=''):
        diff = np.abs((val - ref) / ref)
        if hasattr(diff, '__iter__'):
            diffmax = diff.max()
        else:
            diffmax = diff
        self.assertLess(diffmax, eps, msg=msg)

    def assertIndexListEqual(self, val, ref):
        self.assertEqual(len(val), len(ref))
        for (a, b) in zip(val, ref):
            self.assertEqual(len(a), len(b))
            self.assertTrue(np.all(a == b))


def generate_suite(test_cls, test_items):
    test_suite = unittest.TestSuite()
    for item in test_items:
        test_suite.addTest(test_cls(item))
    return test_suite

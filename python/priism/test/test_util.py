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
import numpy

import priism.alma.util as util
from . import utils as testutils


class RandomIndexGeneratorTest(unittest.TestCase):
    """
    Test RandomIndexGenerator

    test_too_few_ws            test num_ws < num_fold (raise AssertionError)
    test_negative_subset_id    test negative subset ID (raise AssertionError)
    test_too_large_subset_id   test subset ID >= num_fold (raise AssertionError)
    test_random_index          test random index for subset
    test_random_index_with_mod test random index for subset (num_ws % num_fold > 0)
    test_default_fold          test default num_fold (should be 10)
    """
    def _init_generator(self, num_ws, num_fold=None):
        if num_fold is None:
            return util.RandomIndexGenerator(num_ws)
        else:
            return util.RandomIndexGenerator(num_ws, num_fold)

    def test_too_few_ws(self):
        """test_too_few_ws: test num_ws < num_fold (raise AssertionError)"""
        with self.assertRaises(AssertionError):
            self._init_generator(3, 10)

    def test_negative_subset_id(self):
        """test_negative_subset_id: test negative subset ID (raise AssertionError)"""
        num_ws = 20
        num_fold = 5
        generator = self._init_generator(num_ws, num_fold)
        with self.assertRaises(AssertionError):
            random_index = generator.get_subset_index(-1)

    def test_too_large_subset_id(self):
        """test_too_large_subset_id: test subset ID >= num_fold (raise AssertionError)"""
        num_ws = 20
        num_fold = 5
        generator = self._init_generator(num_ws, num_fold)
        with self.assertRaises(AssertionError):
            random_index = generator.get_subset_index(num_fold)
        with self.assertRaises(AssertionError):
            random_index = generator.get_subset_index(num_fold + 1)

    def _run_successful_test(self, num_ws, num_fold=None):
        generator = self._init_generator(num_ws, num_fold)

        # global consistency check
        random_index = generator.random_index
        self.assertEqual(num_ws, len(random_index))
        index_flag = numpy.zeros(num_ws, dtype=numpy.bool)
        self.assertTrue(numpy.all(index_flag == False))
        delta_list = []

        # per subset check
        nmod = num_ws % num_fold
        ndiv = num_ws / num_fold
        for subset_id in range(num_fold):
            subset_index = generator.get_subset_index(subset_id)
            print('subset {0}: index {1}'.format(subset_id, subset_index))

            # check if size of subset index is correct
            num_index = ndiv
            if subset_id < nmod:
                num_index += 1
            self.assertEqual(num_index, len(subset_index))

            # check if subset index is sorted in ascending order
            if num_index > 1:
                delta = subset_index[1:] - subset_index[:-1]
                self.assertTrue(numpy.all(delta > 0))

            # check if subset index is random (not regularly spaced)
            if num_index > 1:
                # NOTE: index can be regularly spaced by chance even if
                #       index assignment is globally random. So, per
                #       subset check is skipped and instead store
                #       deltas for each subset for subsequent use
                delta = subset_index[1:] - subset_index[:-1]
                delta_list.append(delta)

            # activate flag
            index_flag[subset_index] = True

        # check if all index are included in any of subset
        self.assertTrue(numpy.all(index_flag == True))

        # check if index spacing is not unique
        print(delta_list)
        flattened = []
        for delta in delta_list:
            flattened.extend(delta)
        self.assertFalse(numpy.all(flattened == flattened[0]))

    def test_random_index(self):
        """test_random_index: test random index for subset"""
        self._run_successful_test(20, 5)

    def test_random_index_with_mod(self):
        """test_random_index: test random index for subset (num_ws % num_fold > 0)"""
        self._run_successful_test(23, 5)

    def test_default_fold(self):
        """test_default_fold: test default num_fold (should be 10)"""
        generator = self._init_generator(100)
        self.assertEqual(10, generator.num_fold)


def suite():
    test_items = ['test_too_few_ws',
                  'test_negative_subset_id',
                  'test_too_large_subset_id',
                  'test_random_index',
                  'test_random_index_with_mod',
                  'test_default_fold']
    test_suite = testutils.generate_suite(RandomIndexGeneratorTest,
                                          test_items)
    return test_suite

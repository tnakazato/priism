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

import numpy


class RandomIndexGenerator(object):
    def __init__(self, num_ws, num_fold=10):
        self.num_ws = num_ws
        self.num_fold = num_fold
        assert self.num_fold <= self.num_ws

        self.random_index = numpy.arange(self.num_ws, dtype=numpy.int64)
        numpy.random.shuffle(self.random_index)

        self.num_subws = numpy.zeros(self.num_fold, dtype=numpy.int64)
        ndiv = self.num_ws // self.num_fold
        nmod = self.num_ws % self.num_fold
        self.num_subws[:] = ndiv
        self.num_subws[:nmod] += 1

        assert self.num_subws.sum() == self.num_ws
        assert self.num_subws.max() - self.num_subws.min() < 2

        # sort a priori
        for subset_id in range(self.num_fold):
            random_index = self.get_subset_index(subset_id)
            random_index.sort()

    def get_subset_index(self, subset_id):
        assert 0 <= subset_id
        assert subset_id < self.num_fold

        num_subws = self.num_subws[subset_id]
        index_start = numpy.sum(self.num_subws[:subset_id])
        index_end = index_start + num_subws
        random_index = self.random_index[index_start:index_end]

        return random_index

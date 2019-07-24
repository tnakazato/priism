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

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

import almasparsemodeling

import numpy as np


def demo():
    # create worker instance
    worker = almasparsemodeling.AlmaSparseModeling()

    # configuration
    vis = 'input.ms'
    worker.selectdata(vis=vis)
    worker.defineimage()

    # grid visibility in uv-plane
    worker.gridvis()

    # run MFISTA on gridded visibility to obtain optimized image
    l1 = 1.0
    lsqtv = 1.0
    worker.mfista(l1, lsqtv)

    # compute CV
    # cv1 is based on raw visibility while
    # cv2 is based on gridded visibility
    cv1 = worker.computecv(num_fold=10)
    cv2 = worker.computegridcv(num_fold=10)

    # evaluate approximate CV
    acv = worker.computeapproximatecv()

    # export image as FITS
    imagename = 'result.fits'
    arr = np.zeros((100, 100, 1, 1), dtype=np.float32)
    worker.imagearray = arr
    worker.exportimage(imagename=imagename, overwrite=True)

    # publish result
    result = almasparsemodeling.AlmaSparseModelingResult(imagename=imagename,
                                                         cv=cv1, acv=acv)
    return result

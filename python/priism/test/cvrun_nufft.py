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
import time

import priism.alma

start_time = time.time()

# instantiate
worker = priism.alma.AlmaSparseModelingImager(solver='mfista_nufft')

# select data
worker.selectdata(vis='cyc3_hd142527_selfcal.ms',
                  spw='0',
                  #intent='OBSERVE_TARGET#ON_SOURCE',
                  datacolumn='data')

# define image
imsize = [128, 128]
cell = ['0.06arcsec', '0.06arcsec']
worker.defineimage(imsize=imsize,
                   cell=cell,
                   phasecenter='0',  # field ID 0
                   nchan=1,
                   start=0,
                   width=1)

# read vis
worker.readvis()

# mfista
L1_list = [1e0, 1e2, 1e4, 1e6, 1e7, 1e8]
Ltsv_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]

# set num_fold to 0 to disable CV
num_fold = 10
worker.crossvalidation(L1_list, Ltsv_list, num_fold=num_fold,
                       imageprefix='hd142527', imagepolicy='full',
                       summarize=True, figfile='cvresult.png',
                       datafile='cvresult.dat', maxiter=1000)

# If you want to obtain the image for a certain combination of L1 and Ltsv,
# these two lines can be useful. You can control max number of iterations
# or threshold value for mfista. See cvrun_core.py for detail.
#worker.solve(l1=1e0, ltsv=1e8)
#worker.exportimage(imagename='myimage.fits')

end_time = time.time()
print('{0}: elapsed {1} sec'.format('cvrun_nufft.py', end_time - start_time))

# export griddedvis for non-CASA test
#worker.griddedvis.exportdata('griddedvis.dat')

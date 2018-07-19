import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import prism.alma
import prism.core.sparseimaging as sparseimaging

start_time = time.time()

# instantiate
worker = prism.alma.AlmaSparseModelingImager(solver_name='sparseimaging')

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
                   phasecenter='0', #field ID 0
                   nchan=1,
                   start=0,
                   width=1)

# configure gridding operation
worker.configuregrid(convsupport=3,
                     convsampling=100,
                     gridfunction='SF')

# grid
worker.gridvis()

# mfista
L1_list = [1e0, 1e2, 1e4, 1e6, 1e7, 1e8]
Ltsv_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]

# set num_fold to 0 to disable CV
num_fold=10
worker.cvforgridvis(L1_list, Ltsv_list, num_fold=num_fold,
                    imageprefix='hd142527', imagepolicy='full',
                    summarize=True, figfile='cvresult.png',
                    datafile='cvresult.dat', maxiter=1000)

end_time = time.time()
print('{0}: elapsed {1} sec'.format('cvrun.py', end_time-start_time))

# export griddedvis for non-CASA test
inputs = sparseimaging.SparseImagingInputs.from_gridder_result(worker.griddedvis)
inputs.export('griddedvis.dat')

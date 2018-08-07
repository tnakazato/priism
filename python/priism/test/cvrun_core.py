import time
try:
    import cPickle as pickle    
except ImportError:
    import pickle

import priism

start_time = time.time()

# instantiate
worker = priism.SparseModelingImager(solver_name='sparseimaging')

# import visibility data
worker.importvis(filename='griddedvis.dat', flipped=False)

# mfista
L1_list = [1e0, 1e2, 1e4, 1e6, 1e7, 1e8]
Ltsv_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]

# set num_fold to 0 to disable CV
num_fold=10
worker.cvforgridvis(L1_list, Ltsv_list, num_fold=num_fold,
                    imageprefix='hd142527_core', imagepolicy='full',
                    summarize=True, figfile='cvresult_core.png',
                    datafile='cvresult_core.dat', maxiter=1000)

# If you want to obtain the image array for a certain combination of L1 and Ltsv, 
# commented out lines below should be useful. The first example is to iterate 
# the loop 50000 times. The second example is to continue the process until 
# the deviation from the previous steps becomes smaller than threshold computed 
# from eps. Finally, output image array is stored in worker.imagearray.
#worker.mfista(l1=1e0, ltsv=1e8, maxiter=50000)
#worker.mfista(l1=1e0, ltsv=1e8, eps=1e-5)
#image = worker.imagearray

end_time = time.time()
print('{0}: elapsed {1} sec'.format('cvrun_core.py', end_time-start_time))

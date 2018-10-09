import time

import priism.alma

start_time = time.time()

# instantiate
worker = priism.alma.AlmaSparseModelingImager(solver='mfista_fft')

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

# If you want to obtain the image for a certain combination of L1 and Ltsv, 
# these two lines can be useful. You can control max number of iterations 
# or threshold value for mfista. See cvrun_core.py for detail.
#worker.mfista(l1=1e0, ltsv=1e8)
#worker.exportimage(imagename='myimage.fits')

end_time = time.time()
print('{0}: elapsed {1} sec'.format('cvrun.py', end_time-start_time))

# export griddedvis for non-CASA test
#worker.griddedvis.exportdata('griddedvis.dat')

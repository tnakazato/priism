import numpy
import almasparsemodeling

# instantiate
libpath = '{0}/lib'.format(aspm_root)
worker = almasparsemodeling.AlmaSparseModeling(external_solver=True,
                                               libpath=libpath)

# select data
worker.selectdata(vis='cyc3_hd142527_selfcal.ms',
                  #spw='0,1,2,3',
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
try:
    worker.gridvis()
except Exception, e:
    print 'ERROR IN GRIDDING STAGE'
    print str(e)
    import traceback
    print traceback.format_exc()
    raise 

# Cross Validation
L1_list = [1e0, 1e2, 1e4, 1e6, 1e8]
Ltsv_list = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]

num_fold = 10
worker.cvforgridvis(L1_list, Ltsv_list, num_fold=num_fold, imageprefix='HD142527', imagepolicy='full',
                    summarize=True, figfile='cvresult.png', datafile='cvresult.dat')


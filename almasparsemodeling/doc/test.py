import numpy
import almasparsemodeling

# start processing
import time
start_time = time.time()

# instantiate
libpath = '{0}/lib'.format(aspm_root)
worker = almasparsemodeling.AlmaSparseModeling(external_solver=True,
                                               libpath=libpath)

# select data
worker.selectdata(vis='cyc3_hd142527_selfcal.ms',
                  #vis='test.ms',
                  spw='0',
                  #intent='OBSERVE_TARGET#ON_SOURCE',
                  datacolumn='data')

# define image
# suggested value
#imsize = [43,43]
#cell = ['0.23arcsec', '0.23arcsec']
imsize = [128, 128]
cell = ['0.06arcsec', '0.06arcsec']
worker.defineimage(imsize=imsize,
                   cell=cell,
                   phasecenter='0', #field ID 0
                   nchan=1,
                   start=0,
                   width=1)

# configure gridding operation
#worker.configuregrid(convsupport=3,
#                     convsampling=100,
#                     gridfunction='SF')
worker.configuregrid(convsupport=1,
                     convsampling=100,
                     gridfunction='BOX')

# grid
try:
    worker.gridvis()
except Exception, e:
    print 'ERROR IN GRIDDING STAGE'
    print str(e)
    import traceback
    print traceback.format_exc()
    raise 

# mfista
L1=0.01
Ltsv=100.0

try:
    worker.mfista(L1, Ltsv)
except Exception, e:
    print 'ERROR IN MFISTA STAGE'
    print str(e)
    import traceback
    print traceback.format_exc()
    raise

# export resulting image as FITS
imagename='HD142527.fits'
try:
    worker.exportimage(imagename, overwrite=True)
except Exception, e:
    print 'ERROR IN EXPORT STAGE'
    print str(e)
    import traceback
    print traceback.format_exc()
    raise

# completed
end_time = time.time()
print 'Process completed. Product is "{0}".'.format(imagename)
print 'Resulting image array has nonzero pixels? {0}'.format(
    numpy.any(worker.imagearray != 0.0))
print 'Elapsed {0} sec'.format(end_time - start_time)

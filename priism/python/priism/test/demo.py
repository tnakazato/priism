from __future__ import absolute_import

import almasparsemodeling 

import numpy

def demo():
    # create worker instance
    worker = almasparsemodeling.AlmaSparseModeling()
    
    # configuration
    vis='input.ms'
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
    arr = numpy.zeros((100,100,1,1),dtype=numpy.float32)
    worker.imagearray = arr
    worker.exportimage(imagename=imagename, overwrite=True)
    
    # publish result
    result = almasparsemodeling.AlmaSparseModelingResult(imagename=imagename,
                                             cv=cv1, acv=acv)
    return result
    
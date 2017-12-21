from __future__ import absolute_import

import numpy

import almasparsemodeling.external.sakura as sakura
import almasparsemodeling.core as core

class AlmaSparseModelingResult(object):
    """
    This is a class to hold a result produced by AlmaSparseModeling
    """
    def __init__(self, imagename, cv=-1.0, acv=-1.0):
        """
        Constructor
        
        Parameters:
            imagename  name of the FITS cube
            cv         associating cross validation
            acv        associating approximate cross validation
        """
        self.imagename = imagename
        self.cv = cv
        self.acv = acv
        
    def __repr__(self):
        return 'imagename "{0}"\n'.format(self.imagename) \
             + '  cross validation           = {0}\n'.format(self.cv) \
             + '  cross validation (approx.) = {0}\n'.format(self.acv)

class AlmaSparseModelingCore(object):
    """
    Core implementation of sparse modeling specialized for ALMA.
    It performs visibility gridding on uv-plane.
    """
    def __init__(self, external_solver=True, libpath=None):
        """
        Constructor
        
        Parameters:
            external_solver use external solver by S. Ikeda. 
                            Default is True since internal solver 
                            is not ready yet.
            libpath         (effective when external_solver is True)
                            library path to external solver
       """
        self.__initialize()
        self.external_solver = external_solver
        self.libpath = libpath
        
    def selectdata(self, vis, field='', spw='', timerange='', uvrange='', antenna='', 
                  scan='', observation='', intent='', datacolumn='corrected'):
        """
        Select visibility data.
        
        Parameters:
            vis             name of measurement set
            field           field selection (default '' ---> all)
            spw             spw selection (default '' ---> all)
            timerange       timerange selection (default '' ---> all)
            uvrange         uvrange selectoin  (default '' ---> all)
            antenna         antenna/baseline selection (default '' ---> all)
            scan            scan selection (default '' ---> all)
            observation     observation ID selection (default '' ---> all)
            intent          intent selection (default '' ---> all)
            datacolumn      data column to be used ('data' or 'corrected')
        """
        visparam = core.ParamContainer.CreateContainer(core.VisParamContainer, **locals())
        self.visparams.append(visparam)
    
    def defineimage(self, imsize=100, cell='1arcsec', phasecenter='', projection='SIN',
                    nchan=-1, start='', width='', outframe='LSRK', stokes='I'):
        """
        Define resulting image.
        
        start, width, and nchan are defined as follows:
        
          start=<center frequency of first image channel>
            |
        |-------|-------|-------| nchan=3
        |<----->|
          width=<constant channel width of image channel>

        
        Parameters:
            imsize          number of pixels for the resulting image 
                            (default 100 ---> [100,100])
            cell            pixel size for the resulting image
                            (default '1arcsec' ---> ['1arcsec', '1arcsec']
            phasecenter     phase center direction or field ID (default '')
            projection      projection scheme (default 'SIN')
            nchan           number of spectral channels
            start           start channel/frequency
            width           width in channel/frequency
            outframe        output frequency reference frame (fixed to 'LSRK')
            stokes          stokes parameter (fixed to 'I')
        """
        self.imparam = core.ParamContainer.CreateContainer(core.ImageParamContainer, 
                                                           **locals())
        self.uvgridconfig = self.imparam.uvgridconfig
    
    def configuregrid(self, convsupport, convsampling, gridfunction):
        if isinstance(gridfunction, str):
            gridfunction = core.GridFunctionUtil.sf(convsupport, convsampling)
        self.gridparam = core.ParamContainer.CreateContainer(core.GridParamContainer, 
                                                             **locals())
    
    def gridvis(self, parallel=False):
        """
        Grid visibility data on uv-plane.
        """
        # gridvis consists of several steps:
        #     1. select and read data according to data selection
        #     2. pre-gridding data processing
        #     3. give the data to gridder
        #     4. post-gridding data processing
        # 
        visgridder = core.VisibilityGridder(self.gridparam, self.imparam)
        
        # workaround for strange behavior of ms iterator
        interval=1.0e-16
        for visparam in self.visparams:
            reader = core.VisibilityReader(visparam)
            converter = core.VisibilityConverter(visparam, self.imparam)
            if parallel:
                for working_set in sakura.paraMap(self.num_threads, 
                                                  converter.generate_working_set, 
                                                  reader.readvis(interval=interval)):
                    visgridder.grid(working_set)
            else:
                for chunk in reader.readvis(interval=interval):
                    working_set = converter.generate_working_set(chunk)
                    visgridder.grid(working_set)
        self.griddedvis = visgridder.get_result()
    
    def mfista(self, l1, ltsv):
        """
        Run MFISTA algorithm on gridded visibility data.
        gridvis must be executed beforehand.
        
        Parameters:
            l1      L1 regularization term
            ltsv   TSV regularization term
        """
        self.mfistaparam = core.ParamContainer.CreateContainer(core.MfistaParamContainer, 
                                                               **locals())
        self.imagearray = self._mfista(self.mfistaparam, self.griddedvis)
    
    def _mfista(self, mfistaparam, griddedvis):
        assert griddedvis is not None
        if not self.external_solver:
            solver = core.MfistaSolver(mfistaparam)
        else:
            # using MFISTA solver by S. Ikeda
            solver = core.MfistaSolverExternal(mfistaparam, libpath=self.libpath)
        return solver.solve(griddedvis)
    
    def __initialize(self):
        # configuration
        self.imparam = None
        self.visparams = []
        
        # working array
        self.griddedvis = None
        self.imagearray = None
        
        # optimize number of threads
        self.num_threads = 2

    
class AlmaSparseModeling(AlmaSparseModelingCore):
    """
    AlmaSparseModeling inherits all core functions from its parent. 
    It additionally equips to compute direct and approximate cross 
    validation of resulting image as well as a function to export 
    resulting image as an FITS cube.
    """
    def __init__(self, external_solver=True, libpath=None):
        """
        Constructor
        
        Parameters:
            external_solver use external solver by S. Ikeda. 
                            Default is True since internal solver 
                            is not ready yet.
            libpath         (effective when external_solver is True)
                            library path to external solver
        """
        super(AlmaSparseModeling, self).__init__(external_solver=external_solver,
                                                 libpath=libpath)
                
    def exportimage(self, imagename, overwrite=False):
        """
        Export MFISTA result as an image (FITS cube).
        mfista must be executed beforehand.
        
        Parameters:
            imagename  name of output image name
        """
        if self.imparam is None:
            raise RuntimeError('You have to define image configuration before export!')
        self.imparam.imagename = imagename
        
        if self.imagearray is None:
            raise RuntimeError('You don\'t have an image array!')
        
        # convert phasecenter if it is given as FIELD_ID
        if isinstance(self.imparam.phasecenter, str) and self.imparam.phasecenter.isdigit():
            print 'Use PHASE_DIR for FIELD {0}'.format(self.imparam.phasecenter)
            # take first MS
            vis = self.visparams[0].vis
            field_id = int(self.imparam.phasecenter)
            phase_direction = core.ImageWriter.phase_direction_for_field(vis=vis, 
                                                                         field_id=field_id)
            self.imparam.phasecenter = phase_direction
        if (isinstance(self.imparam.start, str) and self.imparam.start.isdigit()) \
            or isinstance(self.imparam.start, int):
            # TODO: we need LSRK frequency
            start = self.imparam.start
            spw = int(self.visparams[0].as_msindex()['spw'][0])
            print 'Use Freuquency for channel {0} spw {1}'.format(start, spw)
            cf, cw = core.ImageWriter.frequency_setup_for_spw(vis=vis, 
                                                              spw_id=spw,
                                                              chan=start)
            self.imparam.start = cf
            self.imparam.width = cw
        imagemeta = core.ImageMetaInfoContainer.fromvis(vis)
        writer = core.ImageWriter(self.imparam, self.imagearray, imagemeta)
        writer.write(overwrite=overwrite)

    def computecv(self, num_fold=10):
        """
        Compute cross validation on resulting image.
        Cross validation is evaluated based on raw visibility 
        (i.e., prior to gridding).
        """
        assert self.griddedvis is not None
        
        evaluator = core.MeanSquareErrorEvaluator()
        visgridder = core.CrossValidationVisibilityGridder(self.gridparam, 
                                                           self.imparam, 
                                                           self.griddedvis.num_ws,
                                                           num_fold)
        
        for i in xrange(num_fold):
            for visparam in self.visparam:
                reader = VisibilityReader(visparam)
                converter = VisibilityConverter(visparam, self.imparam)
                for working_set in sakura.paraMap(self.num_threads, 
                                           converter.generate_working_set, 
                                           reader.readvis()):
                    visgridder.grid(working_set, i)
            griddedvis = visgridder.get_result()
            visibility_cache = visgridder.get_visibility_cache()
            
            # run MFISTA
            imagearray = self._mfista(self.mfistaparam,
                                      griddedvis)
            
            # evaluate MSE (Mean Square Error)
            mse = evaluator.evaluate_and_accumulate(visibility_cache,
                                                    imagearray)
            
        mean_mse = evaluator.get_mean_mse()
        
        return mean_mse
    
    def computegridcv(self, l1, ltsv, num_fold=10):
        """
        Compute cross validation on resulting image.
        Cross validation is evaluated based on gridded visibility.
        """
        mfistaparam = core.ParamContainer.CreateContainer(core.MfistaParamContainer, 
                                                          l1=l1, ltsv=ltsv)
        assert self.griddedvis is not None
        real_data = self.griddedvis.real
        active = numpy.flatnonzero(real_data)
        num_active = len(active)
        
        evaluator = core.MeanSquareErrorEvaluator()
        subset_handler = core.GriddedVisibilitySubsetHandler(self.griddedvis, 
                                                           self.uvgridconfig, 
                                                           num_fold)
        
        for i in xrange(num_fold):
            # pick up subset for cross validation
            subset_handler.generate_subset(subset_id=i)
            
            try:            
                # run MFISTA
                imagearray = self._mfista(mfistaparam, 
                                          subset_handler.visibility_active)

                # evaluate MSE (Mean Square Error)
                mse = evaluator.evaluate_and_accumulate(subset_handler.visibility_cache, 
                                                        imagearray,
                                                        self.uvgridconfig)

            finally:
                # restore zero-ed pixels
                subset_handler.restore_visibility()
             
        mean_mse = evaluator.get_mean_mse()
        
        return mean_mse
    
    def computeapproximatecv(self):
        """
        Evaluate approximate cross validation on resulting image.
        """
        assert griddedvis is not None
        evaluator = core.ApproximateCrossValidationEvaluator()
        
        acv = evaluator.evaluate(self.griddedvis)
        return 0.0


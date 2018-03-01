from __future__ import absolute_import

import os
import shutil
import math
import numpy
import collections
import pylab as pl
import matplotlib
import time

import almasparsemodeling.external.sakura as sakura
import almasparsemodeling.core as core
import almasparsemodeling.external.casa as casa

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
        self.external_solver = external_solver
        self.libpath = libpath
        self.__initialize()
        
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
    
    def mfista(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None, 
               storeinitialimage=True, overwriteinitialimage=False):
        """
        Run MFISTA algorithm on gridded visibility data.
        gridvis must be executed beforehand.
        
        Parameters:
            l1 -- L1 regularization term
            ltsv -- TSV regularization term
            maxiter -- maximum number of iteration for MFISTA
            eps -- threshold factor for MFISTA
            clean_box -- clean box as a float array
            storeinitialimage -- keep the result as an initial image for next run
            overwriteinitialimage -- overwrite existing initial image
        """
        self.mfistaparam = core.ParamContainer.CreateContainer(core.MfistaParamContainer, 
                                                               l1=l1, ltsv=ltsv,
                                                               maxiter=maxiter, eps=eps,
                                                               clean_box=clean_box)
        self.imagearray = self._mfista(self.mfistaparam, self.griddedvis,
                                       storeinitialimage=storeinitialimage, overwriteinitialimage=overwriteinitialimage)
    
    def _mfista(self, mfistaparam, griddedvis, storeinitialimage=True, overwriteinitialimage=False):
        assert griddedvis is not None
#         if not self.external_solver:
#             solver = core.MfistaSolver(mfistaparam)
#         else:
#             # using MFISTA solver by S. Ikeda
#             solver = core.MfistaSolverExternal(mfistaparam, libpath=self.libpath)
        self.solver.mfistaparam = mfistaparam
        return self.solver.solve(griddedvis, storeinitialimage, overwriteinitialimage)
    
    def __initialize(self):
        # configuration
        self.imparam = None
        self.visparams = []
        
        # working array
        self.griddedvis = None
        self.imagearray = None
        
        # TODO: optimize number of threads
        self.num_threads = 2
        
        # create MFISTA instance with dummy parameter
        mfistaparam = core.ParamContainer.CreateContainer(core.MfistaParamContainer, 
                                                          l1=0.0, ltsv=0.0)
        if not self.external_solver:
            self.solver = core.MfistaSolver(mfistaparam)
        else:
            self.solver = core.MfistaSolverExternal(mfistaparam, libpath=self.libpath)

    
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
        vis = self.visparams[0].vis
        if isinstance(self.imparam.phasecenter, str) and self.imparam.phasecenter.isdigit():
            print 'Use PHASE_DIR for FIELD {0}'.format(self.imparam.phasecenter)
            # take first MS
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
        
    def cvforgridvis(self, l1_list, ltsv_list, num_fold=10, imageprefix='image', imagepolicy='full', 
               summarize=True, figfile=None, datafile=None, maxiter=50000, eps=1.0e-5, clean_box=None):
        """
        Perform cross validation and search the best parameter for L1 and Ltsv from 
        the given list of these.
        
        Inputs:
            l1_list -- list of L1 values to examine
            ltsv_list -- List of Ltsv values to examine
            num_fold -- number of visibility subsets for cross validation
            imageprefix -- prefix for output image
                           imageprefix is used for the best image (<imageprefix>.fits) 
            imagepolicy -- policy of output image ('full' or 'best')
                           full: keep all FITS image produced by cross validation
                           best: only keep FITS image corersponding to the best solution
            summarize -- generate summary plot if True
            figfile -- name of summary figure of cross validation. 
                       None will not produce a file.
            datafile -- name of output data file containing whole MSE values. 
                        None will not produce a file.
            maxiter -- maximum number of iteration for MFISTA algorithm
            eps -- threshold factor for MFISTA algorithm
            clean_box -- clean box as a float array (default None)
        
        Output:
            dictionary containing best L1 (key: L1), best Ltsv (key;Ltsv), and
            corresponding image name (key: image, should be <imageprefix>.fits)
        """
        start_time = time.time()
        
        # sanity check
        if imagepolicy not in ('best', 'full'):
            raise ArgumentError('imagepolicy must be best or full. {0} was provided.'.format(imagepolicy))
        
        try:
            np_l1_list = numpy.asarray(l1_list)
            np_ltsv_list = numpy.asarray(ltsv_list)
        except Exception, e:
            print 'Exception occurred'
            print str(e)
            raise ArgumentError('l1_list or ltsv_list (or both) seems invalid.')
        
        if str(np_l1_list.dtype) == 'object':
            raise ArgumentError('l1_list contains invalid value')
        if str(np_ltsv_list.dtype) == 'object':
            raise ArgumentError('ltsv_list contains invalid value')
        
        result_L1 = []
        result_Ltsv = []
        result_mse = []
        result_image = []
        
        num_L1 = len(np_l1_list)
        num_Ltsv = len(np_ltsv_list)
        L1_sort_index = numpy.argsort(np_l1_list)
        Ltsv_sort_index = numpy.argsort(np_ltsv_list)
        
        sorted_l1_list = np_l1_list[L1_sort_index]
        sorted_ltsv_list = np_ltsv_list[Ltsv_sort_index]
        
        if summarize is True:
            PlotterClass = CVPlotter
        else:
            PlotterClass = NullPlotter
        plotter = PlotterClass(num_L1, num_Ltsv, sorted_l1_list, sorted_ltsv_list)
            
        if datafile is not None:
            f = open('cvresult.dat', 'w')
        else:
            f = open(os.devnull, 'w')
        print >> f, '# L1, Ltsv, MSE'
        
        # initialize CV
        self.initializecv(num_fold=num_fold)
                
        #for j, Ltsv in enumerate(np_ltsv_list):
        # loop Ltsv in ascending order
        for j in xrange(num_Ltsv):
            Ltsv = sorted_ltsv_list[j]
            # trick to update initial image when Ltsv is changed
            overwrite_initial = True
            
            #for i, L1 in enumerate(np_l1_list):
            # loop L1 in descending order
            for i in xrange(num_L1 - 1, -1, -1):
                L1 = sorted_l1_list[i]
                result_L1.append(L1)
                result_Ltsv.append(Ltsv)
                
                # get full visibility image first
                imagename = 'L1_{0}_Ltsv_{1}.fits'.format(int(math.log10(L1)), int(math.log10(Ltsv)))
                self.mfista(L1, Ltsv, maxiter=maxiter, eps=eps, clean_box=clean_box,
                            storeinitialimage=True, overwriteinitialimage=overwrite_initial)
                self.exportimage(imagename, overwrite=True)
                result_image.append(imagename)
                
                # then evaluate MSE
                mse = self.computegridmse(L1, Ltsv, maxiter, eps, clean_box)
                result_mse.append(mse)
                
                print 'L1 10^{0} Ltsv 10^{1}: MSE {2} FITS {3}'.format(int(math.log10(L1)),
                                                                       int(math.log10(Ltsv)),
                                                                       mse,
                                                                       imagename)
                print >> f, '{0}, {1}, {2}'.format(L1, Ltsv, mse)
        
                if summarize is True:
                    with casa.OpenImage(imagename) as ia:
                        chunk = ia.getchunk()
                    data = numpy.squeeze(chunk) # data will be 2D
                    
                    plotter.plotimage(i, j, data, mse)
                    
                # As long as Ltsv is kept, initial image will not be updated
                #overwrite_initial = False
                            
        # finalize CV
        self.finalizecv()
        
        f.close()
        
        best_solution = numpy.argmin(result_mse)
        best_mse = result_mse[best_solution]
        best_image = result_image[best_solution]
        best_L1 = result_L1[best_solution]
        best_Ltsv = result_Ltsv[best_solution]
        
#         L1_index = np_l1_list.tolist().index(best_L1)
#         Ltsv_index = np_ltsv_list.tolist().index(best_Ltsv)
        L1_index = numpy.where(sorted_l1_list == best_L1)[0][0]
        Ltsv_index = numpy.where(sorted_ltsv_list == best_Ltsv)[0][0]
        if best_mse >= 0.0:
            plotter.mark_bestimage(L1_index, Ltsv_index)
        
        plotter.draw()
        if figfile is not None:
            plotter.savefig(figfile)
        # completed
        end_time = time.time()
        
        if best_mse >= 0.0:
            print 'Process completed. Optimal result is as follows'
            print '    L1, Ltsv = 10^{0}, 10^{1}'.format(int(math.log10(best_L1)), int(math.log10(best_Ltsv)))
            print '    MSE = {0}'.format(best_mse)
            print '    imagename = {0}'.format(best_image)
        else:
            print 'Process completed. Cross-validation was not performed.'
            print 'WARNING: Optimal solution will not be correct one since no CV was executed.'
        
        print 'Elapsed {0} sec'.format(end_time - start_time)
        
        
        # copy the best image to final image
        shutil.copy2(best_image, imageprefix+'.fits')
        if imagepolicy == 'full':
            # keep all intermediate images
            pass
        elif imagepolicy == 'best':
            # remove all intermediate images
            for imagename in result_image:
                os.remove(imagename)
        else:
            assert False
            
        # finally, return best L1 and Ltsv
        return {'L1': best_L1, 'Ltsv': best_Ltsv}
           

    def initializecv(self, num_fold=10):
        assert self.griddedvis is not None
        
        if (not hasattr(self, 'visset')) or self.visset is None:
            self.visset = core.VisibilitySubsetGenerator(self.griddedvis, num_fold) 
    
    def finalizecv(self):
        self.visset = None
    
    def computegridmse(self, l1, ltsv, maxiter=50000, eps=1.0e-5, clean_box=None):
        """
        Compute mean-square-error (MSE) on resulting image.
        MSE is evaluated from gridded visibility.
        """
        mfistaparam = core.ParamContainer.CreateContainer(core.MfistaParamContainer, 
                                                          l1=l1, ltsv=ltsv,
                                                          maxiter=maxiter, eps=eps, 
                                                          clean_box=clean_box)
        assert self.griddedvis is not None
        
        evaluator = core.MeanSquareErrorEvaluator()
        num_fold = self.visset.num_fold
        
        if num_fold <= 1:
            # CV is disabled
            return -1.0
        
        subset_handler = core.GriddedVisibilitySubsetHandler(self.visset, 
                                                             self.uvgridconfig)
        
        for i in xrange(num_fold):
            # pick up subset for cross validation
            with subset_handler.generate_subset(subset_id=i) as subset:
            
                # run MFISTA
                imagearray = self._mfista(mfistaparam, 
                                          subset.visibility_active,
                                          False, False)

                # evaluate MSE (Mean Square Error)
                mse = evaluator.evaluate_and_accumulate(subset.visibility_cache, 
                                                        imagearray,
                                                        self.uvgridconfig)
             
        mean_mse = evaluator.get_mean_mse()
        
        return mean_mse
    
    def computemse(self, num_fold=10):
        """
        Compute mean-square-error (MSE) on resulting image.
        MSE is evaluated from raw visibility (i.e., prior to gridding).
        """
        raise NotImplementedError('Computation of MSE from raw visibility is future development.')
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
    
    def computeapproximatemse(self):
        """
        Evaluate approximate mean-square-error (MSE) on resulting image.
        """
        raise NotImplementedError('Computation of Approximate MSE (LOOE) is not implemented yet.')
        assert griddedvis is not None
        evaluator = core.ApproximateCrossValidationEvaluator()
        
        acv = evaluator.evaluate(self.griddedvis)
        return 0.0


class CVPlotter(object):
    def __init__(self, nv, nh, L1_list, Ltsv_list):
        self.nh = nh
        self.nv = nv

        self.left_margin = 0.1
        self.right_margin = 0.1
        self.bottom_margin = 0.1
        self.top_margin = 0.1
        total_width = 1.0 - (self.left_margin + self.right_margin)
        total_height = 1.0 - (self.bottom_margin + self.top_margin)
        dx = total_width / float(self.nh)
        dy = total_height / float(self.nv)
        self.dx = min(dx, dy)
        self.dy = dx
        f = pl.figure(num='CVPlot', figsize=(8,8))
        pl.clf()
        left = self.left_margin
        bottom = self.bottom_margin
        height = self.dy * self.nv
        width = self.dx * self.nh
        outer_frame = pl.axes([left, bottom, width, height])
        outer_frame.set_xlim(-0.5, self.nh - 0.5)
        outer_frame.set_ylim(-0.5, self.nv - 0.5)
        outer_frame.set_xlabel('log10(Ltsv)')
        outer_frame.set_ylabel('log10(L1)')
        outer_frame.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(self.nh)))
        outer_frame.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(self.nv)))
        outer_frame.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: int(math.log10(Ltsv_list[int(x)]))))
        outer_frame.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: int(math.log10(L1_list[int(x)]))))
        
        self.L1_list = L1_list
        self.Ltsv_list = Ltsv_list
        
        self.axes_list = collections.defaultdict(dict)
        
    def plotimage(self, row, column, data, mse):
        left = self.left_margin + column * self.dx
        bottom = self.bottom_margin + row * self.dy
        height = self.dx
        width = self.dy
        #print 'pl.axes([{0}, {1}, {2}, {3}])'.format(left, bottom, width, height)
        nx, ny = data.shape
        a = pl.axes([left, bottom, width, height])
        a.imshow(numpy.flipud(data.transpose()))
        if mse >= 0.0:
            a.text(nx-2, 5, '{:.5g}'.format(mse), ha='right', va='top', fontdict={'size': 'small', 'color': 'white'})
        a.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        a.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        self.axes_list[row][column] = a

    def mark_bestimage(self, row, column):
        best_axes = self.axes_list[row][column]
        bbox = best_axes.get_position()
        best_frame = pl.axes(bbox, axisbg='none')
        best_frame.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        best_frame.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        for loc, spine in best_frame.spines.iteritems():
            spine.set_color('red')
            spine.set_linewidth(3)
            
    def draw(self):
        pl.draw()
        
    def savefig(self, figfile):
        pl.savefig(figfile)

        
class NullPlotter(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def plotimage(self, *args, **kwargs):
        pass
        
    def mark_bestimage(self, *args, **kwargs):
        pass

    def draw(self, *args, **kwargs):
        pass
        
    def savefig(self, *args, **kwargs):
        pass
    

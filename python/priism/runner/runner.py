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

version = '2023-2-27'

import os
import sys
import priism.alma 


class Session: 
    """
    This is a class to run sparse model imaging package PRIISM. 
    """
    def __init__(self, vis=False, field=False, spw=False, ch=False, 
                 datacolumn='data', imname=False, imsize=False, cell=False, 
                 nchan=1, start=0, width=1, 
                 cvname=False, solver='mfista_nufft', l1=1e0, ltsv=1e8, 
                 maxiter=1000, numFold=10, imagePolicy='full', 
                 l1List=[1e0, 1e5, 1e10], ltsvList=[1e0, 1e6, 1e12],
                 optimizer='classical', bayesopt_maxiter=15):

        """
        Constructor 
        
        Parameters: 
          vis         Input MS data of visibility 
          field       Field ID 
          spw         SPW ID
          ch          Channel 
          datacolumn  Data column to be used 
          imname      Output image name
          imsize      Image size in pixel 
          cell        Pixel size 
          nchan       Number of channels to be imaged
          start       Starting channel to be imaged
          width       Channel width for image cube
          cvname      File name prefix for cross validation 
          solver      Solver for sparse modeling 
          l1          L1 parameter for sparse modeling
          ltsv        Ltsv parameter for sparse modeling
          maxiter     Max iteration parameter for sparse modeling
          numFold     Number of fold parameter for sparse modeling
          imagePolicy 'full' to save all images or 'best' to save only the best
          l1List      List of L1 for cross validation
          ltsvList    List of Ltsv for cross validation
          optimizer   Method for cross validation, 'classical' or 'bayesian'
          bayesopt_maxiter  Max iteration for Bayesian optimizer
        """

        # Parameter dictionary
        self.p = {} 
        # Data selection parameters
        self.p['vis'] = vis
        self.p['field'] = field
        self.p['spw'] = spw
        self.p['ch'] = ch
        self.p['datacolumn'] = datacolumn 

        # Image parameters 
        self.p['imname'] = imname
        self.p['imsize'] = imsize
        self.p['cell'] = cell
        self.p['nchan'] = 1
        self.p['start'] = 0
        self.p['width'] = 1

        # Sparse Model parameters 
        self.p['cvname'] = cvname
        self.p['solver'] = solver
        self.p['l1'] = l1
        self.p['ltsv'] = ltsv
        self.p['maxiter'] = maxiter 
        self.p['numFold'] = numFold
        self.p['imagePolicy'] = imagePolicy
        self.p['l1List'] = l1List
        self.p['ltsvList'] = ltsvList
        self.p['optimizer'] = optimizer 
        self.p['bayesopt_maxiter'] = bayesopt_maxiter

        # Vis file check
        print('#############################')
        if type(vis) == bool:
            raise TypeError("MS file should be specified with 'vis' parameter to start a session!")

        if os.path.exists(vis):
            print('(runner.Session): vis = '+vis)
        else:
            raise FileNotFoundError(vis)


    def saveParam(self, paramFile, overwrite=False):
        """
        Save parameters to a file. 
        """
        while os.path.exists(paramFile) and not overwrite:
            print('Parameter file '+paramFile+' alrady exists. Enter different file name. ')

            paramFile = ''
            while not paramFile: # to avoid empty name
                paramFile = input('## Enter parameter file name: ')

        print('## Parameters are seved to '+paramFile)

        with open(paramFile,'w') as pf:
            print("# Parameter file for PRIISM runner.py", file=pf)
            kys = self.p.keys()
            for k in kys:
                # formatting with tab
                if len(k) >= 7:
                    out = str(k)+':\t'+str(self.p[k])
                else:
                    out = str(k)+':\t \t'+str(self.p[k])
                print(out, file=pf)
        pass


    @classmethod
    def loadParam(cls, paramFile):
        """
        Load parameters from a file and return Session instance
        """

        if os.path.exists(paramFile):
            print('## Reading parameter file : '+paramFile)
        else:
            raise FileNotFoundError(paramFile)
        
        with open(paramFile, 'r') as pf:
            ptxts = pf.readlines()

        p = {}
        for ptxt in ptxts:
            isNull = not ptxt.strip() # Null lines
            isComment = False 
            if not isNull:
               if ptxt.strip()[0] == '#': # skip comment
                   isComment = True 

            if not isNull and not isComment:
                # To remove line feed, tab and space
                table = str.maketrans({'\n': '','\t': '', ' ': '',})
                a  = ptxt.translate(table).split(':')
                if len(a) != 2:
                    raise ValueError('Separator ":" should be used to specify a parameter.')
                else:
                    if a[0] in ['field','spw','ch','imsize','cell','nchan','start','width','l1','ltsv','maxiter','numFold','l1List','ltsvList','bayesopt_maxiter']: # non str param
                        if a[0] == 'ch' and '~' in a[1]:
                            p[a[0]] = a[1]
                        else:
                            p[a[0]] = eval(a[1]) # convert str 
                    else:
                        if a[1] == 'False': a[1]=False
                        p[a[0]] = a[1]

        session = cls(**p)
        
        return session

    def setDataParam(self, field=False, spw=False, ch=False, datacolumn=False):
        """
        Set parameters for data selection
        """
        if not type(field) == bool:
            self.p['field'] = field

        if not type(spw) == bool:
            self.p['spw'] = spw

        if not type(ch) == bool:
            self.p['ch'] = ch

        if not type(datacolumn) == bool:
            self.p['datacolumn'] = datacolumn

        pass


    def setImageParam(self, imname=False, imsize=False, cell=False, 
                      nchan=False, start=False, width=False):
        """
        Set parameters for imaging
        """
        if not type(imsize) == bool: 
            self.p['imsize'] = imsize

        if not type(cell) == bool: 
            self.p['cell'] = cell

        if not type(imname) == bool: 
            self.p['imname'] = imname

        if not type(nchan) == bool: 
            self.p['nchan'] = nchan

        if not type(start) == bool: 
            self.p['start'] = start

        if not type(width) == bool: 
            self.p['width'] = width

        pass


    def setSpParam(self,cvname=False, solver=False, l1=False, ltsv=False, maxiter=False, numFold = False, l1List=False, ltsvList=False, imagePolicy=False, optimizer=False, bayesopt_maxiter=False):
        """
        Set parameters for sparse modeling. Default values are given 
        in __init__(). 
        """ 

        if not type(cvname) == bool:
            self.p['cvname'] = cvname

        if not type(solver) == bool: 
            if solver in ['mfista_fft', 'mfista_nufft']:
                self.p['solver'] = solver
            else:
                print('## No such solver, '+solver+'. Set to default solver: mfista_nufft')

        if not type(l1) == bool:
            self.p['l1'] = l1

        if not type(ltsv) == bool:
            self.p['ltsv'] = ltsv
        
        if not type(maxiter) == bool:
            self.p['maxiter'] = maxiter

        if not type(numFold) == bool:
            self.p['numFold'] = numFold

        if not type(l1List) == bool:
            self.p['l1List'] = l1List

        if not type(ltsvList) == bool:
            self.p['ltsvList'] = ltsvList
            
        if not type(imagePolicy) == bool:
            if imagePolicy in ['full', 'best']:
                self.p['imagePolicy'] = imagePolicy
            else:
                print('## No such imagePolicy, '+imagePolicy+'. Set to default policy: "full"')

        if not type(optimizer) == bool:
            if optimizer in ['classical', 'bayesian']:
                self.p['optimizer'] = optimizer
            else:
                print('## No such optimizer, '+optimizer+'. Set to default optimizer: "classical"')

        if not type(bayesopt_maxiter) == bool:
            self.p['bayesopt_maxiter'] = bayesopt_maxiter

        pass


    def checkParam(self):
        """
        Check parameters 
        """
        
        print(self.p)

        pass


    def _setWorker(self):
        """
        Internal method for preparation of sparse modeling 
        """
        print('###### Start PRIISM')        
        self.worker = priism.alma.AlmaSparseModelingImager(solver=self.p['solver'])

        ###### print(type(self.p['ch']) )
        if type(self.p['ch']) == bool:
            spwch = str(self.p['spw'])
        else:
            spwch = str(self.p['spw'])+':'+str(self.p['ch'])

        self.worker.selectdata(vis=self.p['vis'], 
                               field = str(self.p['field']),
                               spw= spwch,
                               intent='OBSERVE_TARGET#ON_SOURCE',
                               datacolumn=self.p['datacolumn'])

        self.worker.defineimage(imsize= self.p['imsize'],
                           cell= self.p['cell'],
                           phasecenter= str(self.p['field']), #field,
                           nchan=int(self.p['nchan']),
                           start=int(self.p['start']),
                           width=int(self.p['width']))

        self.worker.readvis()
        pass

    def run(self, overwrite=False): 
        """
        Run for sparse modeling image
        """
        # load imager instance 
        try:
            self.worker 
        except AttributeError:
            try: 
                self._setWorker()
            except Exception as err:
                print('  ')
                print("###### Failed to start PRIISM at run(). Please check parameters and data. ", str(err))
                print('  ')
                raise err

        # Fix image name 
        if not self.p['imname']:
            self.p['imname'] = input('## Enter image file name (FITS): ')

        print('## Image name : '+self.p['imname'] )

        # File check for overwrite=False
        while os.path.exists(self.p['imname']) and not overwrite:
            print('Output image alrady exists. Enter different image name. ')

            imname = ''
            while not imname: # to avoid empty name
                imname = input('## Enter image file name (FITS): ')

            # Add '.fits' if not there
            if len(imname.split('fits')) == 1:
                self.p['imname'] = imname+'.fits'
            else:
                self.p['imname'] = imname

        # solve + exportimage took about 40 sec 
        self.worker.solve(l1=float(self.p['l1']), ltsv=float(self.p['ltsv']), 
                          maxiter=int(self.p['maxiter']), 
                          storeinitialimage=False, scalehyperparam=False)
        self.worker.exportimage(imagename=self.p['imname'], overwrite=overwrite)
            
        pass

    
    def crossValidation(self, overwrite=False):
        """
        Start cross validation 
        """
        try:
            self.worker 
        except AttributeError:
            try: 
                self._setWorker()
            except Exception as err:
                print('  ')
                print("###### Failed to start PRIISM at crossValidation(). Please check parameters and data. ", str(err))
                print('  ')
                raise err


        # Fix output file name 
        if not self.p['cvname']:
            while not self.p['cvname']: # to avoid empty name
                self.p['cvname'] = input('## Enter prefix for CV outputs: ')

        # File name check for overwrite=False
        while os.path.exists(self.p['cvname']) and not overwrite: 
            print('Output directory for CV, '+self.p['cvname']+', exists. Enter different cvname. ')
            cvname = ''
            while not cvname: # to avoid empty name
                cvname = input('## Enter prefix for CV outputs: ')
            self.p['cvname'] = cvname
                

        print('## File name prefix for cross validation: '+self.p['cvname'] )
        
        self.worker.crossvalidation(self.p['l1List'], self.p['ltsvList'], 
                        num_fold= int(self.p['numFold']),
                        imageprefix=self.p['cvname'], 
                        imagepolicy=self.p['imagePolicy'],
                        summarize=True, 
                        figfile=self.p['cvname']+'.cvresult.png',
                        datafile=self.p['cvname']+'.cvresult.dat', 
                        maxiter=int(self.p['maxiter']),
                        resultasinitialimage=False, scalehyperparam=False,
                        optimizer=self.p['optimizer'],
                        bayesopt_maxiter=int(self.p['bayesopt_maxiter']))

        
        # Moving output files to cvname directory 
        if not os.path.exists(self.p['cvname']):
            os.system('mkdir '+self.p['cvname'])
        os.system('mv '+self.p['cvname']+'.cvresult.*'+' '+self.p['cvname'])
        os.system('mv '+self.p['cvname']+'.fits'+' '+self.p['cvname'])
        if self.p['imagePolicy'] == 'full':
            os.system('mv '+self.p['cvname']+'_L1*fits'+' '+self.p['cvname'])
        # Save parameter file
        self.saveParam('%s/%s.param'%(self.p['cvname'],self.p['cvname']),overwrite=overwrite)

        pass
    
if __name__ == '__main__':

    vis = 'twhya_smoothed_scan12.ms'
    h =Session(vis, field=0, spw=0, ch=24)
    h.setImageParam(imname='myimage.fits',imsize=[256,256], cell=['0.08arcsec'])
    h.checkParam()
    h.saveParam('a.param')
    h.run()

    h.setSpParam(cvname='cvtest')
    h.crossValidation()

    # Run using a parameter file
    hh = Session.loadParam('a.param')
    hh.crossValidation(overwrite=True)

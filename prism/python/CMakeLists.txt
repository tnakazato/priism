cmake_minimum_required(VERSION 2.8)

message(STATUS "Entering prism subdirectory...")

install(FILES prism/__init__.py
        DESTINATION prism)
        
install(FILES 
        prism/core/__init__.py
        prism/core/cv.py
        prism/core/datacontainer.py
        prism/core/imager.py
        prism/core/mfista.py
        prism/core/paramcontainer.py
        prism/core/sparseimaging.py
        prism/core/util.py
        DESTINATION prism/core)
        
install(FILES
        prism/alma/__init__.py
        prism/alma/gridder.py
        prism/alma/imager.py
        prism/alma/imagewriter.py
        prism/alma/paramcontainer.py
        prism/alma/util.py
        prism/alma/visconverter.py
        prism/alma/visreader.py
        DESTINATION prism/alma)
        
install(FILES prism/external/__init__.py
        DESTINATION prism/external)
        
install(FILES
        prism/external/casa/__init__.py
        prism/external/casa/casa_atexit.py
        prism/external/casa/casatools.py
        DESTINATION prism/external/casa)
        
install(FILES
        prism/external/sakura/__init__.py
        prism/external/sakura/allocator.py
        prism/external/sakura/core.py
        prism/external/sakura/reductionhelper.py
        DESTINATION prism/external/sakura)

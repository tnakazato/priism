cmake_minimum_required(VERSION 2.8)

message(STATUS "Entering priism subdirectory...")

install(FILES priism/__init__.py
        DESTINATION priism)
        
install(FILES 
        priism/core/__init__.py
        priism/core/cv.py
        priism/core/datacontainer.py
        priism/core/imager.py
        priism/core/mfista.py
        priism/core/paramcontainer.py
        priism/core/sparseimaging.py
        priism/core/util.py
        DESTINATION priism/core)
        
install(FILES
        priism/alma/__init__.py
        priism/alma/gridder.py
        priism/alma/imager.py
        priism/alma/imagewriter.py
        priism/alma/paramcontainer.py
        priism/alma/util.py
        priism/alma/visconverter.py
        priism/alma/visreader.py
        DESTINATION priism/alma)
        
install(FILES priism/external/__init__.py
        DESTINATION priism/external)
        
install(FILES
        priism/external/casa/__init__.py
        priism/external/casa/casa_atexit.py
        priism/external/casa/casatools.py
        DESTINATION priism/external/casa)
        
install(FILES
        priism/external/sakura/__init__.py
        priism/external/sakura/allocator.py
        priism/external/sakura/core.py
        priism/external/sakura/reductionhelper.py
        DESTINATION priism/external/sakura)

install(FILES
        priism/test/__init__.py
        #priism/test/demo.py
        priism/test/cvrun.py
        priism/test/cvrun_core.py
        priism/test/runtest.py
        priism/test/test_gridder.py
        priism/test/test_imagewriter.py
        priism/test/test_util.py
        priism/test/test_visconverter.py
        priism/test/test_visreader.py
        priism/test/utils.py
        DESTINATION priism/test)
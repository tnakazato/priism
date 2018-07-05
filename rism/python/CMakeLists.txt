cmake_minimum_required(VERSION 2.8)

message(STATUS "Entering rism subdirectory...")

install(FILES rism/__init__.py
        DESTINATION rism)
        
install(FILES 
        rism/core/__init__.py
        rism/core/cv.py
        rism/core/datacontainer.py
        rism/core/imager.py
        rism/core/mfista.py
        rism/core/paramcontainer.py
        rism/core/sparseimaging.py
        rism/core/util.py
        DESTINATION rism/core)
        
install(FILES
        rism/alma/__init__.py
        rism/alma/gridder.py
        rism/alma/imager.py
        rism/alma/imagewriter.py
        rism/alma/paramcontainer.py
        rism/alma/util.py
        rism/alma/visconverter.py
        rism/alma/visreader.py
        DESTINATION rism/alma)
        
install(FILES rism/external/__init__.py
        DESTINATION rism/external)
        
install(FILES
        rism/external/casa/__init__.py
        rism/external/casa/casa_atexit.py
        rism/external/casa/casatools.py
        DESTINATION rism/external/casa)
        
install(FILES
        rism/external/sakura/__init__.py
        rism/external/sakura/allocator.py
        rism/external/sakura/core.py
        rism/external/sakura/reductionhelper.py
        DESTINATION rism/external/sakura)

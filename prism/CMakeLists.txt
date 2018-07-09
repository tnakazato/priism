cmake_minimum_required(VERSION 2.8)

project(prism)

# options
option(BUNDLE_CASA_APP "Bundle python-binding to CASA release package (OSX only)" OFF)

# download sparseimaging
set(SPARSEIMAGING_DIR sparseimaging)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR})
  set(SPARSEIMAGING_EXTRACT sparseimaging-development)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_EXTRACT})
    set(SPARSEIMAGING_ZIP development.zip)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/${SPARSEIMAGING_ZIP})
      message(STATUS "Sparseimaging is not downloaded yet")
      set(SPARSEIMAGING_URL https://github.com/ikeda46/sparseimaging/archive/${SPARSEIMAGING_ZIP})
      execute_process(COMMAND curl -L -O ${SPARSEIMAGING_URL}
                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
    execute_process(COMMAND unzip ${CMAKE_BINARY_DIR}/${SPARSEIMAGING_ZIP}
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  endif()
  execute_process(COMMAND ln -s "${SPARSEIMAGING_EXTRACT}" "${SPARSEIMAGING_DIR}"
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

# download sakura
set(SAKURA_DIR libsakura)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR})
  set(SAKURA_TAR libsakura-5.0.0.tgz)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_TAR})
    message(STATUS "Sakura is not downloaded yet")
    set(SAKURA_URL https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/${SAKURA_TAR})
    execute_process(COMMAND curl -L -O ${SAKURA_URL}
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
  execute_process(COMMAND tar zxvf ${CMAKE_BINARY_DIR}/${SAKURA_TAR}
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

message(STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

# download googletest
set(GTEST_DIR gtest)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/${GTEST_DIR})
  set(GTEST_EXTRACT googletest-master)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/${GTEST_EXTRACT})
     set(GTEST_ZIP master.zip)
     if(NOT EXISTS ${CMAKE_BINARY_DIR}/${GTEST_ZIP})
       message(STATUS "GoogleTest is not downloaded yet")
       set(GTEST_URL https://github.com/google/googletest/archive/${GTEST_ZIP})
       execute_process(COMMAND curl -L -O ${GTEST_URL}
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
     endif()
     execute_process(COMMAND unzip ${CMAKE_BINARY_DIR}/${GTEST_ZIP}
                     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${SAKURA_DIR})
  endif()
  execute_process(COMMAND ln -s "${GTEST_EXTRACT}/googletest" "${GTEST_DIR}"
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${SAKURA_DIR})
endif()  

# edit makefile for sparseimaging if necessary
set(SPARSEIMAGING_BUILD_DIR ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR}/c)
message(STATUS "OPENBLAS_LIBRARY_DIR=${OPENBLAS_LIBRARY_DIR}")
if(OPENBLAS_LIBRARY_DIR)
  message(STATUS "Edit makefile for sparseimaging")
  execute_process(COMMAND sed -e "s@-lopenblas@-L${OPENBLAS_LIBRARY_DIR} -lopenblas@g" makefile
                  COMMAND sed -e "s@-Xlinker /usr/lib@-Xlinker ${OPENBLAS_LIBRARY_DIR}@g"
                  OUTPUT_FILE makefile.prism
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
else()
  execute_process(COMMAND cp -f makefile makefile.prism
                  WORKIND_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
endif()

# configure compile of sparseimaging to generate libmfista_fft.so
add_custom_target(sparseimaging make -f makefile.prism libmfista_fft
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})


# configure sakura
#set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/cmake-modules ${CMAKE_MODULE_PATH}")
#add_subdirectory(libsakura)
set(SAKURA_BUILD_DIR ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/build)
if(NOT EXISTS ${SAKURA_BUILD_DIR})
  execute_process(COMMAND mkdir ${SAKURA_BUILD_DIR})
endif()
set(SAKURA_CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_MODULE_PATH=../cmake-modules)
if(EIGEN3_INCLUDE_DIR)
  set(SAKURA_CMAKE_OPTIONS ${SAKURA_CMAKE_OPTIONS} -DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR})
endif()
if(PYTHON_ROOTDIR)
  set(SAKURA_CMAKE_OPTIONS ${SAKURA_CMAKE_OPTIONS} -DPYTHON_ROOTDIR=${PYTHON_ROOTDIR}) 
endif()
if(BUNDLE_CASA_APP)
  set(SAKURA_CMAKE_OPTIONS ${SAKURA_CMAKE_OPTIONS} -DBUNDLE_CASA_APP=ON)
endif() 
message(STATUS "SAKURA_CMAKE_OPTIONS=${SAKURA_CMAKE_OPTIONS}")
execute_process(COMMAND cmake .. ${SAKURA_CMAKE_OPTIONS}
                WORKING_DIRECTORY ${SAKURA_BUILD_DIR})
add_custom_target(sakura make 
                  COMMAND make apidoc
                  WORKING_DIRECTORY ${SAKURA_BUILD_DIR})
                

# add prism module
add_subdirectory(python)
add_custom_target(prism ALL echo ""
                  DEPENDS sakura sparseimaging)

# install libmfista_fft.so
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_fft.so
        DESTINATION prism/core)

# install sakura (currently separated process)
add_custom_target(install_sakura make install/fast
                  WORKING_DIRECTORY ${SAKURA_BUILD_DIR})


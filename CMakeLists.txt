# Copyright (C) 2019
# National Astronomical Observatory of Japan
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
cmake_minimum_required(VERSION 2.8)

project(priism)

# download sparseimaging
set(SPARSEIMAGING_DIR sparseimaging)
execute_process(COMMAND which git
                RESULT_VARIABLE GIT_ERROR_CODE
                OUTPUT_VARIABLE GIT_CMD)
message(STATUS "Found git: ${GIT_CMD}")
if(NOT GIT_ERROR_CODE)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR})
    message(STATUS "Sparseimaging is not downloaded yet")
    set(SPARSEIMAGING_URL https://github.com/ikeda46/sparseimaging.git)
    execute_process(COMMAND git clone ${SPARSEIMAGING_URL}
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    # make sure we are correct branch
    execute_process(COMMAND git checkout smili
                    WORKING_DIRECTORY ${SPARSEIMAGING_DIR})
  endif()
else()
  set(SPARSEIMAGING_EXTRACT sparseimaging-smili)
  set(SPARSEIMAGING_ZIP smili.zip)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR})
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_EXTRACT})
      if(NOT EXISTS ${CMAKE_BINARY_DIR}/${SPARSEIMAGING_ZIP})
        message(STATUS "Sparseimaging is not downloaded yet")
        set(SPARSEIMAGING_URL https://github.com/ikeda46/sparseimaging/archive/${SPARSEIMAGING_ZIP})
        execute_process(COMMAND curl -L -O ${SPARSEIMAGING_URL}
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      endif()
      message(STATUS "Extracting sparseimaging code")
      execute_process(COMMAND unzip ${CMAKE_BINARY_DIR}/${SPARSEIMAGING_ZIP}
                      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
    message(STATUS "Creating ${SPARSEIMAGING_DIR}")
    execute_process(COMMAND ln -s "${SPARSEIMAGING_EXTRACT}" "${SPARSEIMAGING_DIR}"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  endif()
endif()

# download sakura
set(SAKURA_DIR libsakura)
set(SAKURA_TAR libsakura-5.0.8.tgz)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR})
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_TAR})
    message(STATUS "Sakura is not downloaded yet")
    set(SAKURA_URL https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/${SAKURA_TAR})
    execute_process(COMMAND curl -L -O ${SAKURA_URL}
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
  execute_process(COMMAND tar zxvf ${PROJECT_SOURCE_DIR}/${SAKURA_TAR}
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

message(STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

# download googletest
set(GTEST_DIR gtest)
set(GTEST_ZIP master.zip)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/${GTEST_DIR})
  set(GTEST_EXTRACT googletest-master)
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/${GTEST_EXTRACT})
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

# configure sakura
set(SAKURA_BUILD_DIR "${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/build")
if (NOT EXISTS ${SAKURA_BUILD_DIR})
  execute_process(COMMAND mkdir -p ${SAKURA_BUILD_DIR})
endif()
set(SAKURA_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/priism/external/sakura")
set(SAKURA_CMAKE_OPTIONS "-DCMAKE_INSTALL_PREFIX=${SAKURA_INSTALL_PREFIX} -DCMAKE_MODULE_PATH=${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/cmake-modules -DBUILD_DOC=OFF -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_INSTALL_RPATH='$ORIGIN/'")
if (EIGEN3_INCLUDE_DIR)
  set(SAKURA_CMAKE_OPTIONS "${SAKURA_CMAKE_OPTIONS} -DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
endif()
if (PYTHON_ROOTDIR) 
  set(SAKURA_CMAKE_OPTIONS "${SAKURA_CMAKE_OPTIONS} -DPYTHON_ROOTDIR=${PYTHON_ROOTDIR}")
endif()
if (PYTHON_INCLUDE_DIR)
  set(SAKURA_CMAKE_OPTIONS "${SAKURA_CMAKE_OPTIONS} -DPYTHON_INCLUDE_PATH=${PYTHON_INCLUDE_DIR}")
endif()
if (PYTHON_LIBRARY)
  set(SAKURA_CMAKE_OPTIONS "${SAKURA_CMAKE_OPTIONS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY}")
endif()
if (NUMPY_INCLUDE_DIR)
  set(SAKURA_CMAKE_OPTIONS "${SAKURA_CMAKE_OPTIONS} -DNUMPY_INCLUDE_DIR=${NUMPY_INCLUDE_DIR}")
endif()
if (BUNDLE_CASA_APP)
  set(SAKURA_CMAKE_OPTIONS "${SAKURA_CMAKE_OPTIONS} -DBUNDLE_CASA_APP=${BUNDLE_CASA_APP}")
endif()
message(STATUS "SAKURA_CMAKE_OPTIONS=${SAKURA_CMAKE_OPTIONS}")
set(SAKURA_CMAKE_COMMAND "${CMAKE_COMMAND} .. ${SAKURA_CMAKE_OPTIONS}")
execute_process(COMMAND echo "execute_process(COMMAND ${SAKURA_CMAKE_COMMAND}\n                WORKING_DIRECTORY ${SAKURA_BUILD_DIR})"
                OUTPUT_FILE build_sakura.cmake
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
#execute_process(COMMAND cmake .. -DCMAKE_INSTALL_PREFIX=/remote/home/nakazato/priism_root/priism/external/sakura -DCMAKE_MODULE_PATH=/work/a/nakazato/git_workspace/priism/libsakura/cmake-modules -DBUILD_DOC=OFF -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-4/root/usr/bin/g++ -DCMAKE_INSTALL_RPATH='$ORIGIN/' -DEIGEN3_INCLUDE_DIR=/remote/home/nakazato/eigen/eigen -DPYTHON_ROOTDIR=/opt/rh/rh-python36/root/usr
#execute_process(COMMAND ${CMAKE_COMMAND} --debug-output .. ${SAKURA_CMAKE_OPTIONS}
#                WORKING_DIRECTORY ${SAKURA_BUILD_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -P build_sakura.cmake)
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/cmake-modules CACHE STRING "List of directories to search for CMake modules")
#set(BUILD_DOC OFF CACHE BOOL "Build Sakura API document")
#add_subdirectory(libsakura EXCLUDE_FROM_ALL)
#if(NOT APPLE)
#  set_target_properties(sakurapy PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
#else()
#  set_target_properties(sakurapy PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/priism/external/sakura")
#endif()

# edit makefile for sparseimaging if necessary
set(SPARSEIMAGING_BUILD_DIR ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR}/c++)
set(SPARSEIMAGING_MAKEFILE makefile.priism)
message(STATUS "OPENBLAS_LIBRARY_DIR=${OPENBLAS_LIBRARY_DIR}")
message(STATUS "FFTW3_INCLUDE_DIR is ${FFTW3_INCLUDE_DIR}")
execute_process(COMMAND echo ${FFTW3_INCLUDE_DIR}
                COMMAND sed -e "s@include@lib64@g"
                COMMAND tr -d "\n"
                OUTPUT_VARIABLE FFTW3_LIBRARY_DIR)
if(NOT EXISTS ${FFTW3_LIBRARY_DIR}/libfftw3.so)
  execute_process(COMMAND echo ${FFTW3_INCLUDE_DIR}
                COMMAND sed -e "s@include@lib@g"
                COMMAND tr -d "\n"
                OUTPUT_VARIABLE FFTW3_LIBRARY_DIR)
endif()
message(STATUS "FFTW3_LIBRARY_DIR is ${FFTW3_LIBRARY_DIR}")
execute_process(COMMAND sed -e "s@CFLAGS=-O2@CFLAGS=-O2 -I${FFTW3_INCLUDE_DIR}@g" makefile
                COMMAND sed -e "s@-lrt @@g"
                COMMAND sed -e "s@^CLIBS_FFTW = @CLIBS_FFTW = -L${FFTW3_LIBRARY_DIR} @g"
                COMMAND sed -e "s@-O2@-O3@g"
                COMMAND sed -e "s@CXX=.*$@CXX=${CMAKE_CXX_COMPILER}@g"
                COMMAND sed -e "s@c++[0-9][0-9]@c++11@g"
                OUTPUT_FILE ${SPARSEIMAGING_MAKEFILE}
                WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
if(EIGEN3_INCLUDE_DIR)
  message(STATUS "Edit makefile for sparseimaging")
  execute_process(COMMAND sed -i -e "s@/usr/include/eigen3@${EIGEN3_INCLUDE_DIR}@g" ${SPARSEIMAGING_MAKEFILE}
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
endif()

# configure compile of sparseimaging to generate libmfista_fft.so
add_custom_target(sparseimaging make -f ${SPARSEIMAGING_MAKEFILE} libraries
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})

add_custom_target(sakurapy make sakurapy
                  WORKING_DIRECTORY ${SAKURA_BUILD_DIR})

# add priism module
add_subdirectory(python)
add_custom_target(priism ALL echo ""
                  DEPENDS sakurapy sparseimaging)

# install libmfista
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_fft.so
        DESTINATION priism/core)
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_nufft.so
        DESTINATION priism/core)

# install libsakura
execute_process(COMMAND echo "execute_process(COMMAND ${CMAKE_COMMAND} --build ${SAKURA_BUILD_DIR} --target install/fast)"
                OUTPUT_FILE sakura_install.cmake
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
install(SCRIPT ${CMAKE_BINARY_DIR}/sakura_install.cmake)

#if (APPLE)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/bin/libsakura.dylib
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/bin/libsakura.5.dylib
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/bin/libsakura.5.0.dylib
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/python-binding/libsakurapy.so
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/python-binding/libsakurapy.0.so
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/python-binding/libsakurapy.0.1.so
#          DESTINATION priism/external/sakura)
#else()
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/bin/libsakura.so
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/bin/libsakura.so.5
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/bin/libsakura.so.5.0
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/python-binding/libsakurapy.so
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/python-binding/libsakurapy.so.0
#          DESTINATION priism/external/sakura)
#  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${SAKURA_DIR}/python-binding/libsakurapy.so.0.1
#          DESTINATION priism/external/sakura)
#endif()

# clean_all
# clean_all will execute "make clean" as well as removing all downloaded files
add_custom_target(clean_all make clean
                  COMMAND rm -rf ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}
                  COMMAND rm -rf ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR}
                  COMMAND rm -rf ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_EXTRACT}
                  COMMAND rm -f  ${CMAKE_BINARY_DIR}/${SAKURA_TAR}
                  COMMAND rm -f  ${CMAKE_BINARY_DIR}/${SPARSEIMAGING_ZIP}
                  COMMAND rm -f  ${CMAKE_BINARY_DIR}/${GTEST_ZIP}
                  COMMAND echo "All downloaded/extracted materials have been removed. You have to re-do cmake to get them.")

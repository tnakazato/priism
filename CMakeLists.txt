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
cmake_minimum_required(VERSION 2.8.12)

project(priism)

option(USE_INTEL_COMPILER
       "set ON to use Intel C++ compiler to build sparseimaging"
       OFF)

message(STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

# configure sakura
# edit configuration files for sakura
set(SAKURA_DIR libsakura)
execute_process(COMMAND sed -i -e "/^add_subdirectory.g*test/d" CMakeLists.txt
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${SAKURA_DIR})
set(SAKURA_SUBDIRECTORIES src python-binding)
foreach(SAKURA_SUBDIR IN ITEMS src python-binding)
  set(_DIR "${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/${SAKURA_SUBDIR}")
  message(STATUS "editing CMakeLists.txt in ${_DIR}")
  execute_process(COMMAND sed -i -e "s@LIBRARY DESTINATION lib@LIBRARY DESTINATION priism/external/sakura@g" CMakeLists.txt
                  WORKING_DIRECTORY ${_DIR})
  execute_process(COMMAND sed -i -e "s@PUBLIC_HEADER DESTINATION include@PUBLIC_HEADER DESTINATION priism/external/sakura/include@g" CMakeLists.txt
                  WORKING_DIRECTORY ${_DIR})
  unset(_DIR)
endforeach()
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/cmake-modules CACHE STRING "List of directories to search for CMake modules")
set(BUILD_DOC OFF CACHE BOOL "Build Sakura API document")
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/${EIGEN_DIR})
if (APPLE)
  set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/priism/external/sakura")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
add_subdirectory(libsakura)

# edit makefile for sparseimaging if necessary
set(SPARSEIMAGING_DIR sparseimaging)
set(SPARSEIMAGING_BUILD_DIR ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR}/c++)
set(SPARSEIMAGING_MAKEFILE makefile.priism)
message(STATUS "FFTW3_INCLUDE_DIR is ${FFTW3_INCLUDE_DIR}")
execute_process(COMMAND echo ${FFTW3_INCLUDE_DIR}
                COMMAND sed -e "s/include/lib64/g"
                COMMAND tr -d "\n"
                OUTPUT_VARIABLE FFTW3_LIBRARY_DIR)
if(NOT EXISTS ${FFTW3_LIBRARY_DIR}/libfftw3.so)
  execute_process(COMMAND echo ${FFTW3_INCLUDE_DIR}
                COMMAND sed -e "s/include/lib/g"
                COMMAND tr -d "\n"
                OUTPUT_VARIABLE FFTW3_LIBRARY_DIR)
endif()
message(STATUS "FFTW3_LIBRARY_DIR is ${FFTW3_LIBRARY_DIR}")
if(USE_INTEL_COMPILER)
  execute_process(COMMAND sed -e "s:^CFLAGS=.*$:CFLAGS=-O3 -ipo -qopenmp -xHost -I${FFTW3_INCLUDE_DIR}:g" makefile
                  COMMAND sed -e "s:-O2 -Wall:-Wall:g"
                  COMMAND sed -e "s:^CXX=.*:CXX=icpx:g"
                  OUTPUT_FILE ${SPARSEIMAGING_MAKEFILE}
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
else()
  execute_process(COMMAND sed -e "s:CFLAGS=-O2:CFLAGS=-O2 -I${FFTW3_INCLUDE_DIR}:g" makefile
                  COMMAND sed -e "s:-lrt:-lstdc++:g"
                  COMMAND sed -e "s:^CLIBS_FFTW = :CLIBS_FFTW = -L${FFTW3_LIBRARY_DIR} :g"
                  COMMAND sed -e "s:-O2:-O3:g"
                  COMMAND sed -e "s:CXX=.*$:CXX=${CMAKE_CXX_COMPILER}:g"
                  COMMAND sed -e "s:c++[0-9][0-9]:c++11:g"
                  OUTPUT_FILE ${SPARSEIMAGING_MAKEFILE}
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
endif()
if(APPLE)
  execute_process(COMMAND sed -i -e "s:-fopenmp:-Xpreprocessor -fopenmp -lomp :g" ${SPARSEIMAGING_MAKEFILE}
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
endif()
if(EIGEN3_INCLUDE_DIR)
  message(STATUS "Edit makefile for sparseimaging")
  execute_process(COMMAND sed -i -e "s@/usr/include/eigen3@${EIGEN3_INCLUDE_DIR}@g" ${SPARSEIMAGING_MAKEFILE}
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
endif()

# configure compile of sparseimaging to generate libmfista_fft.so
add_custom_target(sparseimaging make clean
                  COMMAND make -f ${SPARSEIMAGING_MAKEFILE} libraries
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})

# add priism module
add_custom_target(priism ALL echo ""
                  DEPENDS sakurapy sparseimaging)

# install libmfista
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_fft.so
        COMPONENT Smili
        DESTINATION priism/core)
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_nufft.so
        COMPONENT Smili
        DESTINATION priism/core)

# clean_all
# clean_all will execute "make clean" as well as removing all downloaded files
add_custom_target(clean_all make clean
                  COMMAND rm -rf ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}
                  COMMAND rm -rf ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR}
                  COMMAND rm -rf ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_EXTRACT}
                  COMMAND rm -f  ${PROJECT_SOURCE_DIR}/${SAKURA_TAR}
                  COMMAND rm -f  ${CMAKE_BINARY_DIR}/${SAKURA_TAR}
                  COMMAND rm -f  ${CMAKE_BINARY_DIR}/${SPARSEIMAGING_ZIP}
                  COMMAND rm -f  ${CMAKE_BINARY_DIR}/${GTEST_ZIP}
                  COMMAND echo "All downloaded/extracted materials have been removed. You have to re-do cmake to get them.")

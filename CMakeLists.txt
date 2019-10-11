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

# download sakura
set(SAKURA_DIR libsakura)
set(SAKURA_TAR libsakura-5.0.7.tgz)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${SAKURA_DIR})
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
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/${SAKURA_DIR}/cmake-modules CACHE STRING "List of directories to search for CMake modules")
set(BUILD_DOC OFF CACHE BOOL "Build Sakura API document")
add_subdirectory(libsakura)
if(NOT APPLE)
  set_target_properties(sakurapy PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
endif()

# edit makefile for sparseimaging if necessary
set(SPARSEIMAGING_BUILD_DIR ${PROJECT_SOURCE_DIR}/${SPARSEIMAGING_DIR}/c)
message(STATUS "OPENBLAS_LIBRARY_DIR=${OPENBLAS_LIBRARY_DIR}")
message(STATUS "FFTW3_INCLUDE_DIR is ${FFTW3_INCLUDE_DIR}")
execute_process(COMMAND echo ${FFTW3_INCLUDE_DIR}
                COMMAND sed -e "s@include@lib@g"
                COMMAND tr -d "\n"
                OUTPUT_VARIABLE FFTW3_LIBRARY_DIR)
message(STATUS "FFTW3_LIBRARY_DIR is ${FFTW3_LIBRARY_DIR}")
if(OPENBLAS_LIBRARY_DIR)
  message(STATUS "Edit makefile for sparseimaging")
  execute_process(COMMAND sed -e "s@-lopenblas@-L${OPENBLAS_LIBRARY_DIR} -lopenblas@g" makefile
                  COMMAND sed -e "s@-Xlinker /usr/lib@-Xlinker ${OPENBLAS_LIBRARY_DIR}@g"
                  COMMAND sed -e "s@^CFLAGS=-O2@CFLAGS=-O2 -I${FFTW3_INCLUDE_DIR}@g"
                  COMMAND sed -e "s@-lrt @@g"
                  COMMAND sed -e "s@^CLIBS_FFTW = @CLIBS_FFTW = -L${FFTW3_LIBRARY_DIR} @g"
                  OUTPUT_FILE makefile.priism
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
else()
  execute_process(COMMAND sed -e "s@CFLAGS=-O2@CFLAGS=-O2 -I${FFTW3_INCLUDE_DIR}@g" makefile
                  COMMAND sed -e "s@-lrt @@g"
                  COMMAND sed -e "s@^CLIBS_FFTW = @CLIBS_FFTW = -L${FFTW3_LIBRARY_DIR} @g"
                  OUTPUT_FILE makefile.priism
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})
endif()

# configure compile of sparseimaging to generate libmfista_fft.so
add_custom_target(sparseimaging make -f makefile.priism libmfista_fft
                  COMMAND make -f makefile.priism libmfista_nufft
                  WORKING_DIRECTORY ${SPARSEIMAGING_BUILD_DIR})             

# add priism module
add_subdirectory(python)
add_custom_target(priism ALL echo ""
                  DEPENDS sakura sparseimaging)

# install libmfista_fft.so
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_fft.so
        DESTINATION priism/core)
install(PROGRAMS ${SPARSEIMAGING_BUILD_DIR}/libmfista_nufft.so
        DESTINATION priism/core)

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

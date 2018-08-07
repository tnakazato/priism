# PRIISM: Python Module for Radio Interferometry Imaging with Sparse Modeling

## Overview

PRIISM is an imaging tool for radio interferometry based on the sparse modeling technique. 
It is implemented as a Python module so that it is able to work on various types of platforms.

PRIISM is not only able to generate an image but also able to choose the best image from the 
range of processing parameters using the cross validation. User can obtain statistically optimal image 
by providing the visibility data with some configuration parameters. 

PRIISM consists of two submodules: `priism.core` and `priism.alma`. The former is generic module 
that provides primitive interface to handle various types of data format while the latter is dedicated 
module for processing ALMA data which is supposed to be working on CASA. 
For `priism.core`, any visibility data must be converted to NumPy array before being fed to the module. 
Output image is also in the form of NumPy array. 
On the other hand, `priism.alma` accepts the visibility data as the MeasurementSet (MS) format, which is 
data format for CASA. `priism.alma` equips some specific interface to handle data in MS format. 
`priism.alma` first performs visibility gridding to put visibility data onto regularly spaced grid in 
uv plane. Then the output image is obtained from gridded visibility using "core" functionality of 
the PRIISM (`priism.core`). 
Finally, the output image, which is NumPy array, is exported as a FITS image with appropriate header 
information. 
By using `priism.alma`, users can directly obtain the FITS image from MS data, and they can immediately 
analyse the result using the applications that support to process FITS images (such as CASA).

PRIISM is simple to use, easy to install. There are two template script that consist of initialization, 
configuration, and processing steps. These are dedicated for `priism.core` and `priism.alma`. 
Scripts are so short that they are within 60 lines including comments and empty lines for readability. 
Users can use these scripts by just editing some lines according to their usage. The scripts should be 
useful to learn how to use PRIISM interactively. 
On install, PRIISM adopts cmake for easy install. As long as prerequisites for PRIISM are fulfilled, 
cmake will configure your build automatically. Also, cmake provides a lot of customization options 
to suits your environment.

It is our hope that PRIISM lower the barrier to entry in the new imaging technique based on the 
sparse modeling. 

## Supported Platform

### `priism (priism.core)`

`priism` should work on any platform fulfilling the prerequisites listed in the Prerequisites section. 

### `priism.alma`

Since `priism.alma` depends on CASA, it should work only on the platforms fulfilling the 
prerequisites for both `priism` and CASA. 

## Tested Platform

### `priism (priism.core)`

`priism` has been tested on the platforms listed below.

* Red Hat Enterprise Linux 6 (RHEL6) with Python 2.7.12
* Red Hat Enterprise Linux 7 (RHEL7) with Python 2.7.12
* Ubuntu 16.04.4 with Python 2.7.12
* Ubuntu 16.04.4 with Python 3.5.2
* macOS 10.12 with Python 2.7.13

### `priism.alma`

`priism.alma` has been tested on the plotforms listed below.

* Red Hat Enterprise Linux 6 (RHEL6) with CASA 5.1.1
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.0.0
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.1.1
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.3.0
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.4.0 (prerelease)
* macOS 10.12 with CASA 5.1.1

## Prerequisites

PRIISM is built using cmake so cmake must be available. 
PRIISM depends on Sakura and sparseimaging library. Sakura is for efficient 
processing and data exchange between Python and C++ layers while the sparseimaging 
library is the heart of the tool, which solves the problem based on the sparse 
modeling. 
Configuration file for PRIISM holds correct versions of these libraries and 
download them during cmake configuration. 
These libraries are compiled and installed with PRIISM.
PRIISM also downloads googletest library, which is required to build Sakura 
library, during cmake configuration. 
These online materials are downloaded by curl command so curl must also be 
available. 

PRIISM for ALMA (`priism.alma`) is supposed to run on CASA. If you want to 
use `priism.alma`, CASA 5.0 or higher must be available.

---
**NOTE**

If network connection is not available, you need to obtain these 
files by yourself and put them to the prescribed location. 
Please see Installing section for detail.

In case if you need to download files by hand, links below might be useful: 
* Sakura library: [https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/libsakura-5.0.0.tgz](https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/libsakura-5.0.0.tgz)
* googletest: [https://github.com/google/googletest/archive/master.zip](https://github.com/google/googletest/archive/master.zip)
* sparseimaging library: [https://github.com/ikeda46/sparseimaging/archive/development.zip](https://github.com/ikeda46/sparseimaging/archive/development.zip)

---

In addition to the dependency on PRIISM, there are several prerequisites from 
Sakura and sparseimaging libraries. Sakura depends on FFTW3 and Eigen3 while 
sparseimaging depends on FFTW3 and OpenBLAS. These libraries should be available.
More importantly, PRIISM requires C++ compiler that supports C++11 features 
since Sakura utilizes various C++11 features in its implementation.

In summary, prerequisites for PRIISM is as follows:

* cmake 2.8 or higher
* curl 
* Python 2.7.x or 3.x (for `priism.core`)
* CASA 5.0 or higher (for `priism.alma`)
* gcc/g++ 4.8 or higher or clang/clang++ 3.5 or higher 
* FFTW3 
* Eigen 3.2 or higher 
* OpenBLAS 


## Installation

### Downloading the Source

Source code of PRIISM can be downloaded from the link below.

[https://alma-intweb.mtk.nao.ac.jp/~nakazato/almasparsemodeling/priism-0.0.0.tgz](https://alma-intweb.mtk.nao.ac.jp/~nakazato/almasparsemodeling/priism-0.0.0.tgz)

Source code is archived and compressed so that it must be extracted using tar command.

    cd priism_root_dir
    tar zxf priism-0.0.0.tgz

In the example above, `priism_root_dir` is an arbitrary directory. 
The directory "priism" will be extracted.

### Building

First, move to the extracted directory and make "build" directory.

    cd priism
    mkdir build
    
Then, cmake command should be run in the "build" directory. In the example below, 
only mandatory option is shown.

    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/priism 
    
Available options for cmake is as follows. Syntax for the option is `-D<name>=<value>`.

**CMAKE_INSTALL_PREFIX**

This option specifies the directory where to install PRIISM.

Example: `-DCMAKE_INSTALL_PREFIX=/usr/local/priism`

**EIGEN3_INCLUDE_DIR**

This option tells cmake where Eigen3 header files are installed. 
This option will be useful when Eigen3 is installed to non-standard location.

Example: `-DEIGEN3_INCLUDE_DIR=$HOME/eigen`

**OPENBLAS_LIBRARY_DIR**

This option tells cmake where OpenBLAS library is installed. 
This option will be useful when OpenBLAS is installed to non-standard location.

Example: `-DOPENBLAS_LIBRARY_DIR=$HOME/OpenBLAS/lib`

**PYTHON_ROOTDIR**

This option tells cmake where python package is installed.
This option will be useful when Python package is installed to non-standard location. 
Typical usage is to link to Python package associating with CASA.

Example: `-DPYTHON_ROOTDIR=/opt/share/casa/Release/casa-release-5.1.1-5.el7`

**BUNDLE_CASA_APP** (macOS only)

This option is a shortcut of `PYTHON_ROOTDIR` specific for macOS. 
If it is set to `ON`, cmake assumes that CASA is installed at standard location (`/Applications/CASA.app`) 
and PRIISM tries to link to Python package associating with CASA.
Note that this option is only effective for macOS.

Example: `-DBUNDLE_CASA_APP=ON`

**Other Options**

There are various customization options for cmake. Please see cmake documentation for detail.

---
**NOTE**

Usually, cmake will download source code for Sakura (+googletest) and sparseimaging. 
If it doesn't happen, you need to obtain the source code from the link listed in the "Prerequisites" section. 
Downloaded files should be put in the "build" directory. For example,

    cd somewhere/network/is/available
    curl -L -O https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/libsakura-5.0.0.tgz
    curl -L -O https://github.com/google/googletest/archive/master.zip
    curl -L -O https://github.com/ikeda46/sparseimaging/archive/development.zip
    cd priism_root_dir/priism/build
    mv somewhere/network/is/available/libsakura-5.0.0.tgz .
    mv somewhere/network/is/available/master.zip
    mv somewhere/network/is/available/development.zip
    
---


### Installing

After you suceed to run cmake, subsequent steps to install PRIISM is just simple.

    make
    make install/fast
    
PRIISM will now be available to the location specified by `CMAKE_INSTALL_PREFIX`. 


## Using PRIISM

### Setting PYTHONPATH
First of all, you need to add following paths to `PYTHONPATH`. 

    <CMAKE_INSTALL_PREFIX>
    <CMAKE_INSTALL_PREFIX>/lib
    
where `<CMAKE_INSTALL_PREFIX>` is the location specified by `CMAKE_INSTALL_PREFIX` when 
cmake is executed. Assuming that `/usr/local/priism` is set for `CMAKE_INSTALL_PREFIX`, 
and we use bash, the command to be executed is as follows.

    export PYTHONPATH=/usr/local/priism:/usr/local/priism/lib:$PYTHONPATH

### Importing module    
Then, launch python or CASA and import appropriate module. For `priism.core`, 

    import priism.core as priism
    
or simply,

    import priism
    
will work. For `priism.alma`, you need to launch CASA. 

    import priism.alma
    
will enable you to use API for ALMA data.

### Template scripts

In the test directory, there are two template scripts that demonstrates how to use PRIISM. 
One is for `priism.core` while the other is for `priism.alma`.


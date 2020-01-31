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
data format for CASA or any python environment (python, ipython, Jupyter Notebook etc.) that has an access to core module of casa, i.e. python environment that can import `casac` module. `priism.alma` equips some specific interface to handle data in MS format.
`priism.alma` first performs visibility gridding to put visibility data onto regularly spaced grid in
uv plane ("FFT" solver). `priism.alma` also has an option to solve raw visibility data ("NUFFT" solver).
Then the output image is obtained from gridded visibility using "core" functionality of
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
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.4.0
* macOS 10.12 with CASA 5.1.1
* Red Hat Enterprise Linux 7 (RHEL7) with python 2.7 plus `casac` module included in CASA 5.4.0

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

PRIISM for ALMA (`priism.alma`) is supposed to run on CASA or python environment with `casac` module. If you want to
use `priism.alma`, CASA 5.0 or higher must be available.

In addition to the dependency on PRIISM, there are several prerequisites from
Sakura and sparseimaging libraries. Sakura depends on FFTW3 and Eigen3 while
sparseimaging depends on FFTW3. These libraries should be available.
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
* git (optional but preferable)


## Installation with `cmake`

### Downloading the Source

You can either clone or download zipped archive of the soruce code from GitHub repository. If you clone the source code, you will get a directory named `priism`
unless you rename it.
If you download zipped source code, you will get a file named `priism-<branch_name>.zip`, which conatains `priism-<branch_name>` as a top-level directory.

---
**NOTE**

If you download priism-0.1.2 or earlier, you will see additional directory layer at the top. More specifically, you will see the following two subdirectories at the top-level directory:

    $ pwd
    priism
    $ ls
    almasparsemodeling  priism

This is older directory structure and almasparsemodeling contains some initial code written at the dawn of the development. In that case, you can build and install priism just translating the directory `priism` into `priism/priism` in the following instruction.

---

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
If network connection is not available, you need to obtain these files by yourself.

In case if you need to download files by hand, links below might be useful:
* Sakura library: [https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/libsakura-5.0.8.tgz](https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/libsakura-5.0.8.tgz)
* googletest: [https://github.com/google/googletest/archive/master.zip](https://github.com/google/googletest/archive/master.zip)
* sparseimaging library: [https://github.com/ikeda46/sparseimaging/archive/smili.zip](https://github.com/ikeda46/sparseimaging/archive/smili.zip)

Downloaded files should be put under the "build" directory. For example,

    cd somewhere/network/is/available
    curl -L -O https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/libsakura-5.0.8.tgz
    curl -L -O https://github.com/google/googletest/archive/master.zip
    curl -L -O https://github.com/ikeda46/sparseimaging/archive/smili.zip
    cd priism_root_dir/priism/build
    mv somewhere/network/is/available/libsakura-5.0.7.tgz .
    mv somewhere/network/is/available/master.zip
    mv somewhere/network/is/available/development.zip

---


### Installing

After you suceed to run cmake, subsequent steps to install PRIISM is just simple.

    make
    make install/fast

PRIISM will now be available to the location specified by `CMAKE_INSTALL_PREFIX`.

## Installation with `setuptools` (Python 3 / CASA 6 only)

As of 0.3.0, PRIISM offers another way of build and install which is based on Python `setuptools`. So far, it simply wraps `cmake` build so the build based on `cmake` is performed underneath. However, it should be easier than `cmake` build especially when you want to install PRIISM to your virtual environment (e.g. the one created by `venv`) because installation directory will automatically be detected by `setuptools` based on which python command is used to run the build and install procedure. Brief instruction on installing PRIISM with modular CASA 6 is shown below.

### 1. Create and Activate Virtual Environment

```
$ python3 -m venv priism
$ source priism/bin/activate
```

### 2. Install CASA 6

Please follow [the instruction](https://casa.nrao.edu/casadocs/casa-5.6.0/introduction/casa6-installation-and-usage) provided by CASA team. You already have a virtual environment for PRIISM so you can use it for installation. 

### 3. Install PRIISM

After moving into the PRIISM's root directory, build and install procedure is as follows:

```
python setup.py build
python setup.py install
```

There are options for those commands similar to `cmake` build. Please see the help for those commands.

```
python setup.py build --help
python setup.py install --help
```

During installation of CASA 6, `numpy` will be installed by dependency. It is recommended to use that `numpy` when you install PRIISM. 

## Using PRIISM

### Setting PYTHONPATH

You have to ensure that the installation directory of the PRIISM is included in the `PYTHONPATH` environment variable. In the case of `cmake` build, installation directory is the location specified by `CMAKE_INSTALL_PREFIX` when cmake is executed. Assuming that `/usr/local/priism` is set for `CMAKE_INSTALL_PREFIX`, and we use bash, the command to be executed is as follows.  

    export PYTHONPATH=/usr/local/priism:$PYTHONPATH

Note that, as of 0.3.0, you no longer need to add `CMAME_INSTALL_PREFIX/lib` to `PYTHONPATH`.

Note also that, in the case of installation with `setuptools`, you usually do not care about `PYTHONPATH` because PRIISM should be installed to the directory that is recognized by default. Otherwise, `setuptools` will notify you to update `PYTHONPATH` when you install PRIISM. 

### Importing module
Then, launch python or CASA and import appropriate module. For `priism.core`,

    import priism.core as priism

or simply,

    import priism

will work. For `priism.alma`, you need to launch CASA or python that can import `casac` module.

    import priism.alma

will enable you to use API for ALMA data.

### Template scripts

In the test directory, there are several template scripts that demonstrates how to use PRIISM.
One is for `priism.core` while the others are for `priism.alma`. There are two versions of
solver: "mfista_fft" and "mfista_nufft". Their usages are bit different so you will find
template scripts for each solver. Name of the scripts are as follows:

* `priism.core` (mfista_fft): `cvrun_core.py`
* `priism.alma` (mfista_fft): `cvrun_fft.py`
* `priism.alma` (mfista_nufft): `cvrun_nufft.py`

### Running `priism.alma` outside CASA

If you want to run `priism.alma` on python environment (i.e. such as python, ipython, Jupyter Notebook, etc.), you have two choices to do that.

1. use modular CASA release (6.0 or later)
1. use `casac` module provided by monolithic CASA release

**The first option is preferable.** But, in case if you have to choose the second option, the following instruction would be useful. 

First, you must be able to import `casac`, which
is a Python binding of CASA toolkit implemented as a set of C++ classes. To do that, you have to do either,

* add a path to `casac.py` to `PYTHONPATH` environment variable before starting python, or
* add the path to `sys.path` after you launch python.

In addition, you have to tell `casac` module where the measures data is located at. There are several ways for this. Here, some examples are shown. The first option is to put the following entry into `~/.casa/rc` file.

    measures.directory: /path/to/casa/measures/data

Note that this is using global configuration file so that it affects to your CASA environment as well as the environment for PRIISM. If this is not desirable, you can localize the scope of the above configuration by using `CASARCFILES` environment variable. First step for localizing the configuration is to create the configuration file that contains the above line. The file should be named or located so that CASA cannot recognize (because it will be globally effective if you name it or put it in the location so that CASA can recognize). One simple way would be to create the configuration file at the current working directory. Then, you can use `CASARCFILES` to tell `casac` module what files it should look for. Here is an example.

    # create configuration file at the working directory
    $ vi .casarc
    $ cat .casarc
    measures.directory: /path/to/casa/measures/data
    # set CASARCFILES environment variable
    $ export CASARCFILES=.casarc

This should be effective only under this directory and will not affect global CASA settings. For more information on the configuration of `casac` module, please see the following link:

http://casacore.github.io/casacore/classcasacore_1_1Aipsrc.html


## License

PRIISM is licensed under GPLv3 as described in COPYING.

## Contact

If you have any questions about PRIISM, please contact Takeshi Nakazato at National Astronomical Observatory of Japan ([@tnakazato](https://github.com/tnakazato) on GitHub).

## Acknowledgement

## Reference

[Nakazato, T., Ikeda, S., Akiyama, K., Kosugi, G., Yamaguchi, M., and Honma, M., 2019, Astronomical Data Analysis Software and Systems XXVIII. ASP Conference Series, Vol. 523, p. 143](http://aspbooks.org/custom/publications/paper/523-0143.html)

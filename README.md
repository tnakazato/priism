# PRIISM: Python Module for Radio Interferometry Imaging with Sparse Modeling

PRIISM is an imaging tool for radio interferometry based on the sparse modeling technique. Here, installation procedure and general description of PRIISM are provided. Please see [Notebook Tutorial](./cvrun.ipynb) on how to use PRIISM. Recommended way to install PRIISM is a combination with CASA 6 modular release. For quick start, please check [Prerequisites](#prerequisites-for-recommended-installation) and then, run the command below. You might want to set up venv for priism.

```
# at top-level directory of priism
$ python3 -m pip install -r requirements.txt
$ python3 setup.py build
$ python3 setup.py install
```

Note that the above installation procedure only works for Linux. If you want to install PRIISM on macOS, or you want to install PRIISM in combination with tar-ball release of CASA, please follow installation instruction below. You can install PRIISM with either [cmake](#installation-with-cmake) or [Python setuptools](#installation-with-setuptools-python-3--casa-6-only) depending on your preference.

<!-- TOC -->

- [Overview](#overview)
- [Supported Platform](#supported-platform)
- [Tested Platform](#tested-platform)
- [Prerequisites](#prerequisites)
- [Installation with `cmake`](#installation-with-cmake)
- [Installation with `setuptools` (Python 3 / CASA 6 only)](#installation-with-setuptools-python-3--casa-6-only)
- [Using PRIISM](#using-priism)
- [License](#license)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Reference](#reference)

<!-- /TOC -->

## Overview

PRIISM is an imaging tool for radio interferometry based on the sparse modeling technique. It is implemented as a Python module so that it is able to work on various types of platforms.

PRIISM is not only able to generate an image but also able to choose the best image from the
range of processing parameters using the cross validation. User can obtain statistically optimal image by providing the visibility data with some configuration parameters.

PRIISM consists of two submodules: `priism.core` and `priism.alma`. The former is generic module that provides primitive interface to handle various types of data format while the latter is dedicated module for processing ALMA data which is supposed to be working on CASA. For `priism.core`, any visibility data must be converted to NumPy array before being fed to the module. Output image is also in the form of NumPy array. On the other hand, `priism.alma` accepts the visibility data as the MeasurementSet (MS) format, which is a native data format for CASA. The `priism.alma` module works on CASA or any python environment (python, ipython, Jupyter Notebook etc.) that has an access to core module of casa tools. `priism.alma` equips some specific interface to handle data in MS format.
Basic workflow is as follows:
1. `priism.alma` first performs visibility gridding to put visibility data onto regularly spaced grid in
uv plane ("FFT" solver). `priism.alma` also has an option to solve raw visibility data ("NUFFT" solver).
1. Next, the output image is obtained from gridded or raw visibility using "core" functionality of
the PRIISM (`priism.core`).
1. Finally, the output image, which is NumPy array, is exported as a FITS image with appropriate header information.

By using `priism.alma`, users can directly obtain the FITS image from MS data, and they can immediately analyse the result using the applications that support to process FITS images (such as CASA).

PRIISM is simple to use, easy to install. Regarding the processing, there are two template script that consist of initialization, configuration, and processing steps. These are dedicated for `priism.core` and `priism.alma`. Scripts are so short that they are within 60 lines including comments and empty lines for readability. Users can use these scripts by just editing some lines according to their usage. The scripts should be useful to learn how to use PRIISM interactively. On install, PRIISM adopts cmake for easy install. As long as prerequisites for PRIISM are fulfilled, cmake will configure your build automatically. Also, cmake provides a lot of customization options to suits your environment. For Python 3 / CASA 6 environment, PRIISM also supports installation based on `setuptools` that wraps cmake build for convenience.

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
* macOS 10.14 with Python 2.7.13

### `priism.alma`

`priism.alma` has been tested on the plotforms listed below.

* Red Hat Enterprise Linux 7 (RHEL7) with CASA 6.0 and 6.1 modular release
* Ubuntu 18.04 with CASA 6.0 and 6.1 modular release
* Red Hat Enterprise Linux 6 (RHEL6) with CASA 5.1.1
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.0.0
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.1.1
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.3.0
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 5.4.0
* macOS 10.14 with CASA 5.1.1
* Red Hat Enterprise Linux 7 (RHEL7) with python 2.7 plus casa tools included in CASA 5.4.0

## Prerequisites

### Prerequisites for Recommended Installation
Prerequisites for an installation wich CASA 6 modular release (recommended) is as follows:

* cmake 2.8 or higher
* curl
* Python 3.6
* gcc/g++ 4.8 or higher or clang/clang++ 3.5 or higher
* FFTW3
* OpenMP 4.0 or higher
* git (optional but preferable)

### Prerequisites for Other Installation

PRIISM provides another ways for installation. They are bit complicated but allow to install PRIISM in more flexible way. Installation on macOS, or installation with CASA tar-ball release should satisfy the following prerequisites.

* cmake 2.8 or higher
* curl
* Python 2.7.x or 3.x (for `priism.core`)
* CASA 5.0 or higher (for `priism.alma`)
* gcc/g++ 4.8 or higher or clang/clang++ 3.5 or higher
* FFTW3
* OpenMP 4.0 or higher
* git (optional but preferable)


## Installation with `cmake`

### Downloading the Source

You can either clone or download zipped archive of the soruce code from GitHub repository. If you clone the source code, you will get a directory named `priism`
unless you rename it.
If you download zipped source code, you will get a file named `priism-<branch_name>.zip`, which conatains `priism-<branch_name>` as a top-level directory.

---
**NOTE**

If you download priism-0.1.2 or earlier, you will see additional directory layer at the top. More specifically, you will see the following two subdirectories at the top-level directory:

```
$ pwd
priism
$ ls
almasparsemodeling  priism
```

This is older directory structure and almasparsemodeling contains some initial code written at the dawn of the development. In that case, you can build and install priism just translating the directory `priism` into `priism/priism` in the following instruction.

---

### Building

First, move to the extracted directory and make "build" directory.

```
cd priism
mkdir build
```

Then, cmake command should be run in the "build" directory. In the example below,
only mandatory option is shown.

    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/priism

Available options for cmake is as follows. Syntax for the option is `-D<name>=<value>`.

**CMAKE_INSTALL_PREFIX**

This option specifies the directory where to install PRIISM.

Example: `-DCMAKE_INSTALL_PREFIX=/usr/local/priism`

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
* Sakura library: [https://alma-intweb.mtk.nao.ac.jp/~sakura/libsakura/libsakura-5.0.8.tgz](https://alma-intweb.mtk.nao.ac.jp/~sakura/libsakura/libsakura-5.0.8.tgz)
* googletest: [https://github.com/google/googletest/archive/master.zip](https://github.com/google/googletest/archive/master.zip)
* sparseimaging library: [https://github.com/ikeda46/sparseimaging/archive/smili.zip](https://github.com/ikeda46/sparseimaging/archive/smili.zip)

Downloaded files should be put under the "build" directory. For example,

    cd somewhere/network/is/available
    curl -L -O https://alma-intweb.mtk.nao.ac.jp/~sakura/libsakura/libsakura-5.0.8.tgz
    curl -L -O https://github.com/google/googletest/archive/master.zip
    curl -L -O https://github.com/ikeda46/sparseimaging/archive/smili.zip
    cd priism_root_dir/priism/build
    mv somewhere/network/is/available/libsakura-5.0.8.tgz .
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

### Create and Activate Virtual Environment

```
$ python3 -m venv priism
$ source priism/bin/activate
```

### Install CASA 6

Please follow [the instruction](https://casa.nrao.edu/casadocs/casa-5.6.0/introduction/casa6-installation-and-usage) provided by CASA team. You already have a virtual environment for PRIISM so you can use it for installation.

### Install PRIISM

After moving to the PRIISM's root directory, build and install procedure is as follows:

```
python setup.py build [any options]
python setup.py install
```

There are options for those commands similar to `cmake` build. Please see the help for detail.

```
python setup.py build --help
python setup.py install --help
```

During installation of CASA 6, `numpy` will be installed by dependency. It is recommended to use that version when you install PRIISM.

### Intel Compiler Support

Intel compiler support (Intel OneAPI) is available. Currently only classic C++ compiler (`icpc`) is supported. After configuring the compiler, the following build option will compile performance critical C++ code with Intel compiler.

```
python setup.py build --use-intel-compiler=yes
```

Note that, due to the incompatibility of Python version, `setvars.sh` should not be used to configure the compiler. Please update `PATH` environment variable manually or use `oneapi/compiler/latest/env/vas.sh` instead.

At runtime, you might need to add `oneapi/intelpython/python3.7/lib` to `LD_LIBRARY_PATH`.

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

will work. For `priism.alma`, you need to launch CASA or python with casa tools.

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

## License

PRIISM is licensed under GPLv3 as described in COPYING.

## Contact

If you have any questions about PRIISM, please contact Takeshi Nakazato at National Astronomical Observatory of Japan ([@tnakazato](https://github.com/tnakazato) on GitHub).

## Acknowledgement

## Reference

* [Nakazato, T., Ikeda, S., Akiyama, K., Kosugi, G., Yamaguchi, M., and Honma, M., 2019, Astronomical Data Analysis Software and Systems XXVIII. ASP Conference Series, Vol. 523, p. 143](http://aspbooks.org/custom/publications/paper/523-0143.html)
* [Nakazato, T. and Ikeda, S., 2020, Astrophysics Source Code Library, record ascl:2006.002](https://ui.adsabs.harvard.edu/abs/2020ascl.soft06002N/abstract)

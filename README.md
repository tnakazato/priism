# PRIISM: Python Module for Radio Interferometry Imaging with Sparse Modeling

PRIISM is an imaging tool for radio interferometry based on the sparse modeling technique. Here, installation procedure and general description of PRIISM are provided. Please see [Notebook Tutorial](./cvrun.ipynb) on how to use PRIISM. Recommended way to install PRIISM is a combination with CASA 6 modular release. For quick start, please check [Prerequisites](#prerequisites-for-recommended-installation) and then, run the command below. You might want to set up venv for priism.

```
# at top-level directory of priism
$ python3 -m pip install -r requirements.txt
$ python3 setup.py build
$ python3 setup.py install
```

<!-- TOC -->

- [Overview](#overview)
- [Supported Platform](#supported-platform)
- [Tested Platform](#tested-platform)
- [Prerequisites](#prerequisites)
- [Installation Procedure in Detail](#installation-procedure-in-detail)
- [Using PRIISM](#using-priism)
- [License](#license)
- [Developer](#developer)
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

PRIISM is simple to use, easy to install. Regarding the processing, there are two template script that consist of initialization, configuration, and processing steps. These are dedicated for `priism.core` and `priism.alma`. Scripts are so short that they are within 60 lines including comments and empty lines for readability. Users can use these scripts by just editing some lines according to their usage. The scripts should be useful to learn how to use PRIISM interactively. We also provide Jupyter Notebook tutorial and demo (see [Using PRIISM](#using-priism) section below). On install, PRIISM adopts installation based on `setuptools` that wraps cmake build for convenience.

It is our hope that PRIISM lower the barrier to entry in the new imaging technique based on the sparse modeling.

## Supported Platform

### `priism (priism.core)`

`priism` should work on any platform fulfilling the prerequisites listed in the Prerequisites section.

### `priism.alma`

Since `priism.alma` depends on CASA, it should work only on the platforms fulfilling the
prerequisites for both `priism` and CASA.

## Tested Platform

### `priism.alma`

`priism.alma` has been tested on the plotforms listed below.

* macOS 11.6.1 with CASA 6.4.0 modular release
* macOS 10.15.7 with CASA 6.5.0 modular release
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 6.5.0 modular release
* Ubuntu 18.04 with CASA 6.5.0 modular release


## Prerequisites

Prerequisites for an installation wich CASA 6 modular release (recommended) is as follows:

* cmake 2.8 or higher
* curl or wget
* Python 3.6 or 3.8
* gcc/g++ 4.8 or higher or clang/clang++ 3.5 or higher
* FFTW3
* OpenMP 4.0 or higher
* git (optional but highly desirable)

## Installation Procedure in Detail

Recommended way to install PRIISM is the use of Python `setuptools` combined with CASA 6 modular release. Installation with monolithic CASA 6 releases is technically possible but is not explained here.

### Create and Activate Virtual Environment

```
$ python3 -m venv priism
$ source priism/bin/activate
```

### Install CASA 6 Modular Release

Please follow [the instruction](https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages) provided by CASA team. You already have a virtual environment for PRIISM so you can use it for installation.

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


 ### Notebook Tutorial/Demo

The following Jupyter Notebook tutorial/demo is available.

 * [Notebook Tutorial (TW Hya)](./cvrun.ipynb)
 * [Notebook Demo (HL Tau)](https://gist.github.com/tnakazato/be0888d153eef2a76a3c260d794bf052)

## License

PRIISM is licensed under GPLv3 as described in COPYING.

## Developer

* Takeshi Nakazato [@tnakazato](https://github.com/tnakazato) - infrastructure, python interface, performance tuning
* Shiro Ikeda [@ikeda46](https://github.com/ikeda46) - algorithm, maintenance of core C++ library, sparseimaging https://github.com/ikeda46/sparseimaging

## Contact

If you have any questions about PRIISM, please contact Takeshi Nakazato at National Astronomical Observatory of Japan ([@tnakazato](https://github.com/tnakazato) on GitHub).

## Acknowledgement

We thank T. Tsukagoshi, M. Yamaguchi, K. Akiyama, and Y. Tamura among many other collaborators.

## Reference

* [Nakazato, T., Ikeda, S., Akiyama, K., Kosugi, G., Yamaguchi, M., and Honma, M., 2019, Astronomical Data Analysis Software and Systems XXVIII. ASP Conference Series, Vol. 523, p. 143](http://aspbooks.org/custom/publications/paper/523-0143.html)
* [Nakazato, T. and Ikeda, S., 2020, Astrophysics Source Code Library, record ascl:2006.002](https://ui.adsabs.harvard.edu/abs/2020ascl.soft06002N/abstract)

# PRIISM: Python Module for Radio Interferometry Imaging with Sparse Modeling

PRIISM is an imaging tool for radio interferometry based on the sparse modeling technique. Here, installation procedure and general description of PRIISM are provided. Please see [Notebook Tutorial](./tutorial_hltau.ipynb,
tutorial_twhya.ipynb) on how to use PRIISM. Recommended way to install PRIISM is a combination with CASA 6 modular release. For quick start, please check [Prerequisites](#prerequisites) and then, run the command below. You might want to set up venv for priism.

> [!IMPORTANT]
> The latest `requirements.txt` intends to support CASA 6.6.4 or higher. For older version of CASA, please use `requirements-old.txt` instead. If you already have PRIISM installation based on CASA 6.6.3 or lower, it is advisable to configure new environment from scratch when you update PRIISM to the latest version, which will be based on CASA 6.6.4 or higher.

> [!TIP]
> If you encountered an error when you install PRIISM on CASA 6.6.4, pleaser first check if you created directory for casadata, `~/.casa/data`. If the directory doesn't exist, please create it.

```
# at top-level directory of priism
$ python3 -m pip install .
```

> [!TIP]
> Since installation with `pip` is still unstable, please use installation with `setup.py`, which is described below, if `pip` installation doesn't work.

Alternatively, self-contained Docker environment is available. The following example launches Jupyter Notebook with PRIISM. In the notebook, `$HOME/work` will be a top-level directory. Access to CLI is also possible. Please see [Docker Environment](#docker-environment) section for detail.

```
# at top-level directory of priism
$ docker compose -f docker/ubuntu/docker-compose.yml up
```

- [Overview](#overview)
- [Supported Platform](#supported-platform)
- [Prerequisites](#prerequisites)
- [Installation Procedure in Detail](#installation-procedure-in-detail)
- [Using PRIISM](#using-priism)
- [License](#license)
- [Developer](#developer)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Reference](#reference)


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

### Tested Platform

`priism.alma` is tested against the following configurations on GitHub Actions.

* Ubuntu 22.04 with CASA 6.6.4 modular release (Python 3.8 and 3.10)
* macOS 13 with CASA 6.6.4 modular release (Python 3.8)
* Ubuntu 22.04 with CASA 6.5.3 monolithic release (Python 3.8)

In addition, it has been manually tested on the following configurations.

* Ubuntu 18.04 with CASA 6.6.4 modular release (Python 3.8)
* Rocky Linux 8.9 with CASA 6.6.4 monolithic release (Python 3.10)


## Prerequisites

Prerequisites for an installation with CASA 6 modular release (recommended) is as follows:

* Python 3.8
* gcc/g++ 4.8 or higher or clang/clang++ 3.5 or higher
* FFTW3
* git (optional but highly desirable)

## Installation Procedure in Detail

Recommended way to install PRIISM is the use of Python pip with CASA 6 modular release. Installation with monolithic CASA 6 releases is technically possible and is briefly explained below.

### Install PRIISM module with CASA 6 modular release

Create and Activate Virtual Environment
```
$ python3 -m venv pyenv/priism
$ source pyenv/priism/bin/activate
```

For the installation CASA 6 Modular Release, please follow [the instruction](https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages) provided by CASA team. You already have a virtual environment for PRIISM so you can use it for installation.

After cloning PRIISM's repository, install procedure is as follows:
```
$ git clone https://github.com/tnakazato/priism.git
  # or get zip file and extract
$ cd priism/
  # install doesn't work with the latest pip
$ python3 -m pip install pip==22.0.4
$ python3 -m pip install .
```
Note that installed `numpy` is not latest one. It is recommended to use that version when you install PRIISM.

Installation with `setup.py` should still work. So, please use the following procedure if installation with `pip` doesn't work.

```
$ python3 -m pip install -r requirements.txt
$ python3 setup.py build
$ python3 setup.py install
```

There are options for those commands. Please see the help for detail.

```
$ python3 setup.py build --help
$ python3 setup.py install --help
```

### Intel Compiler Support

Intel compiler support (Intel OneAPI) is available. Currently only classic C++ compiler (`icpc`) is supported. After configuring the compiler, the following build option will compile performance critical C++ code with Intel compiler.

```
$ export USE_INTEL_COMPILER=yes
$ export LD_LIBRARY_PATH=(PATH in your environment)
$ python3 -m pip install .
```
or

```
$ python3 -m pip install -r requirements.txt
$ python setup.py build --use-intel-compiler=yes
$ python3 setup.py install
```

Note that, due to the incompatibility of Python version, `setvars.sh` should not be used to configure the compiler. Please update `PATH` environment variable manually or use `oneapi/compiler/latest/env/vas.sh` instead.

At runtime, you might need to add `oneapi/intelpython/python3.9/lib` to `LD_LIBRARY_PATH`.


### Install PRIISM for CASA environment

Download and install CASA.
```
$ mkdir ${CASA_DIR}
$ cd ${CASA_DIR}/
$ wget https://casa.nrao.edu/download/distro/casa/release/rhel/casa-6.5.3-28-py3.8.tar.xz --no-check-certificate
$ tar -xvf casa-6.5.3-28-py3.8.tar.xz
$ export PATH=${CASA_DIR}/casa-6.5.3-28-py3.8/bin/:${PATH}
```

Install PRIISM package. You can use CASA's python3 as if it is virtual environment. PRIISM will be installed into `site-packages` directory in CASA.
```
$ cd ${PRIISM_DIR}
$ git clone https://github.com/tnakazato/priism.git
$ cd priism/
$ python3.8 -m pip install --upgrade pip
$ python3.8 -m pip install .
```
You can immediatelly import `priism` from CASA.
```
$ casa --nologger --nogui
CASA<>: import priism
```
### Install for MDAS System at NAOJ

PRIISM nodule can be used in [MDAS system at NAOJ](https://www.adc.nao.ac.jp/MDAS/mdas_e.html) (Multi-wavelength Data Analysis System at National Astronomical Observatory of Japan). See the document [PRIISM-ADC-install_ja.md](./PRIISM-ADC-install_ja.md) (in Japanese) which is part of PRIISM source code. English version is in preparation.

### Docker Environment

As explained above, you can launch Jupyter Notebook with PRIISM using the following command.

```
$ docker compose -f docker/ubuntu/docker-compose.yml up
```

By default, `$HOME/work` will be used for the top-level directory of Jupyter Notebook. If you want to use another directory, you should edit `docker-compose.yml` accordingly.

You may want to enter the container and do something interactively. In that case, you should use other docker commands. If you open another terminal and run `docker ps`, you will see the container named `ubuntu-priism-1` or something similar.

```
$ docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED         STATUS          PORTS                    NAMES
48c94ed59565   ubuntu-priism   "sh -c 'cd /home/ano…"   9 minutes ago   Up 21 seconds   0.0.0.0:8888->8888/tcp   ubuntu-priism-1
```

When you shut down Jupyter Notebook, underlying container is stopped as well. Output of `docker ps` may be empty or may not contain the container for PRIISM. In that case, you need to run container manually through `docker start`.

```
$ docker start ubuntu-priism-1
ubuntu-priism-1
```

If you have running container, you can enter the container with `docker exec`. Below is an example to run `bash` shell inside `ubuntu-priism-1`.

```
$ docker exec -it ubuntu-priism-1 bash
anonymous@48c94ed59565:~$
```

You are also able to build Docker image and run it manually with `docker build` and `docker run`.

```
# build image named "priism-ubuntu"
$ docker build docker/ubuntu -t priism-ubuntu

# run container
$ docker run -it -v $HOME/work:/home/anonymous/work priism-ubuntu

# launch ipython inside container
anonymous@bde5fd3774fc:~$ ipython

# import PRIISM
In [1]: import priism
```


## Using PRIISM

### Importing module
Then, launch python or CASA and import appropriate module. For `priism.core`,
```
>>> import priism.core as priism
```
or simply,
```
>>> import priism
```
will work. For `priism.alma`, you need to launch CASA or python with casa tools.
```
>>> import priism.alma
```
will enable you to use API for ALMA data.

### Template scripts

In the test directory, there are several template scripts that demonstrates how to use PRIISM.
One is for `priism.core` while the others are for `priism.alma`. There are two versions of
solver: "mfista_fft" and "mfista_nufft". Their usages are bit different so you will find
template scripts for each solver. Name of the scripts are as follows:

* `priism.core` (mfista_fft): `cvrun_core.py`
* `priism.alma` (mfista_fft): `cvrun_fft.py`
* `priism.alma` (mfista_nufft): `cvrun_nufft.py`

### Batch Processing
`runner` module is prepared for batch processing using PRIISM module.
Following commands obtain image for the specified MS (measurement set) data (visibility data). Input parameter is (field, SPW, channel) for input MS data and image/pixel size for output image. Parameters are set typical values as default.

```
>>> from priism.runner import runner
>>> vis = 'twhya_smoothed.ms'
>>> imname = 'twhya_smoothed.fits'
>>> h =runner.Session(vis)
>>> h.setDataParam(field=0, spw=0, ch=24)
>>> h.setImageParam(imname=imname, imsize=[256,256], cell=['0.08arcsec'])
>>> h.saveParam('test.param')
>>> h.run()
>>> h.crossValidation()
```

### Notebook Tutorial

The following Jupyter Notebook tutorial/demo is available. TW Hya notebook explains how to run PRIISM interactively while HL Tau notebook is a demo for batch mode.

 * [Jupyter Notebook Tutorial (TW Hya)](./tutorial_twhya.ipynb)
 * [Jupyter Notebook Tutorial (HL Tau)](./tutorial_hltau.ipynb)

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

EOF

# Document for Installing PRIISM: Python Module for Radio Interferometry Imaging with Sparse Modeling

PRIISM is an imaging tool for radio interferometry based on the sparse modeling technique. Here, installation procedure and general description of PRIISM are provided. Please see [Notebook Tutorial](./cvrun.ipynb) on how to use PRIISM. Recommended way to install PRIISM is a combination with CASA 6 modular release. For quick start, please check [Prerequisites](#prerequisites) and then, run the command below. You might want to set up venv for priism.

```
# at top-level directory of priism
$ python3 -m pip install .
```

- [PRIISM Modules](#priism-modules)
- [Tested Platform](#tested-platform)
- [Prerequisites](#prerequisites)
- [Installation Procedure in Detail](#installation-procedure-in-detail)

## PRIISM Modules

### `priism (priism.core)`

`priism` should work on any platform fulfilling the prerequisites listed in the Prerequisites section.

### `priism.alma`

Since `priism.alma` depends on CASA, it should work only on the platforms fulfilling the
prerequisites for both `priism` and CASA.

#	# Tested Platform

`priism.alma` has been tested on the plotforms listed below.

* Red Hat Enterprise Linux 7 (RHEL7) with CASA 6.5.3 modular release
* Red Hat Enterprise Linux 7 (RHEL7) with CASA 6.5.3 modular release in NAOJ/ADC/MDAS system
* CentOS 7 with CASA 6.5.3 modular release
* Ubuntu 18.04 with CASA 6.5.3 modular release
* Ubuntu 20.04 with CASA 6.5.3 modular release
* macOS 10.15 with CASA 6.5.3 modular release


## Prerequisites

Prerequisites for an installation with CASA 6 modular release (recommended) is as follows:

* Python 3.8
* gcc/g++ 4.8 or higher or clang/clang++ 3.5 or higher
* FFTW3
* git (optional but highly desirable)

# Installation Procedure in Detail

Recommended way to install PRIISM is the use of Python pip with CASA 6 modular release. Installation with monolithic CASA 6 releases is technically possible but is not explained here.

## Install PRIISM module with CASA 6 modular release

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
$ python -m pip install pip --upgrade
$ python -m pip install .
```

During installation of CASA 6, `numpy` will be installed by dependency. It is recommended to use that version when you install PRIISM. 


### Intel Compiler Support

Intel compiler support (Intel OneAPI) is available. Currently only classic C++ compiler (`icpc`) is supported. After configuring the compiler, the following build option will compile performance critical C++ code with Intel compiler.

```
$ export USE_INTEL_COMPILER=yes
$ export LD_LIBRARY_PATH=....
$ python -m pip install .
```

Note that, due to the incompatibility of Python version, `setvars.sh` should not be used to configure the compiler. Please update `PATH` environment variable manually or use `oneapi/compiler/latest/env/vas.sh` instead.

At runtime, you might need to add `oneapi/intelpython/python3.9/lib` to `LD_LIBRARY_PATH`. For the NAOJ/ADC/MDAS system, you have to add `/usr/local/gcc/12.2/lib64`. 


## Install PRIISM for CASA environment	

Download and install CASA. 
```
$ mkdir ${CASA_DIR}
$ cd ${CASA_DIR}/
$ wget https://casa.nrao.edu/download/distro/casa/release/rhel/casa-6.5.3-28-py3.8.tar.xz --no-check-certificate
$ tar -xvf casa-6.5.3-28-py3.8.tar.xz
$ export PATH=${CASA_DIR}/casa-6.5.3-28-py3.8/bin/:${PATH}
```

Install PRIISM package. 
```
$ cd ${PRIISM_DIR}
$ git clone https://github.com/tnakazato/priism.git
$ cd priism/
$ python3.8 -m pip install --upgrade pip
$ python3.8 -m pip install .
```

Use PRIISM module as following. You might specify the PATH for the PRIISM module, depending the your installation, 
```
$ casa --nologger --nogui
CASA> import sys
CASA> sys.path.append('${HOME}/.local/lib/python3.8/site-packages')
CASA> import priism
```
Please replace the PATH(`${CASA_DIR}, ${PRIISM_DIR}, ${HOME}`) as your environment. 

### Install PRIISM for NAOJ/ADC/MDAS System

It is better to use Intel compiler. 
```
$ export USE_INTEL_COMPILER=yes
$ export LD_LIBRARY_PATH=/usr/local/gcc/12.2/lib64:${LD_LIBRARY_PATH}
$ python -m pip install .
```
If using Intel compiler, it is necessary to set `LD_LIBRARY_PATH`

## Install with setup.py with options (old version)

```
$ python3 -m pip install pip --upgrade
$ python3 -m pip install -r requirements.txt
$ python3 setup.py build
$ python3 setup.py install
```

There are options for those commands similar to `cmake` build. Please see the help for detail.

```
$ python setup.py build --help
$ python setup.py install --help
```

If you want to use Intel compiler, run with option. 
```
$ python setup.py build --use-intel-compiler=yes
```

EOF

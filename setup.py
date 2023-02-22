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
import io
import os
import shlex
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request as request
import zipfile

from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.command.config import config
from distutils.sysconfig import get_python_inc, get_python_version
from setuptools import setup, find_packages, Command

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def _get_version():
    cwd = os.path.dirname(__file__)
    cwd = cwd if len(cwd) > 0 else '.'
    version_file = os.path.join(cwd, 'python/priism/core/version.py')
    with open(version_file, 'r') as f:
        lines = f.readlines()
    version_line = filter(lambda x: x.startswith('__version__'), lines)
    try:
        version = next(version_line).strip('\n').split('=')[1].strip(" '")
    except StopIteration:
        version = '0.0.0'
    return version

def _requires_from_file(filename):
    cmd_pymod = "python3 -m pip install Cython"
    run_cmd   = subprocess.call(cmd_pymod.split())
    print(run_cmd)
    return open(filename).read().splitlines()

class PriismDependencyError(FileNotFoundError):
    def __init__(self, msg, *args, **kwargs):
        self.msg = msg

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.msg)


def check_command_availability(cmd):
    if isinstance(cmd, list):
        return [check_command_availability(_cmd) for _cmd in cmd]
    else:
        assert isinstance(cmd, str)
        return subprocess.call(['which', cmd], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) == 0


IS_GIT_OK = check_command_availability('git')


def execute_command(cmdstring, cwd=None):
    retcode = subprocess.call(shlex.split(cmdstring), cwd=cwd)
    if retcode != 0:
        print('WARNING: command "{}" failed to execute'.format(cmdstring))
    return retcode


def download_extract(url, filetype):
    req = request.urlopen(url)
    bstream = io.BytesIO(req.read())
    if filetype == 'zip':
        with zipfile.ZipFile(bstream, mode='r') as zf:
            zf.extractall()
    elif filetype == 'tar':
        with tarfile.open(mode='r', fileobj=bstream) as tf:
            tf.extractall()


def opt2attr(s):
    return s[0].strip('=').replace('-', '_')


def opt2env(s):
    return "PRIISM_" + opt2attr(s).upper()


def debug_print_user_options(cmd):
    print('Command: {}'.format(cmd.__class__.__name__))
    print('User Options:')
    for option in cmd.user_options:
        attrname = opt2attr(option)
        attrvalue = getattr(cmd, attrname)
        print('  {}{}'.format(option[0], attrvalue))


def arg_for_set_undefined_options(cmd):
    return tuple((opt, opt) for opt in map(opt2attr, cmd.user_options))


def initialize_attr_for_user_options(cmd):
    for option in cmd.user_options:
        attrname = opt2attr(option)
        setattr(cmd, attrname, None)


def overwrite_attr_for_user_options_by_environ(cmd):
    for option in cmd.user_options:
        attrname = opt2attr(option)
        envname = opt2env(option)
        setattr(cmd, attrname, os.environ.get(envname))


def get_python_library(include_dir):
    libnames = []
    libname1 = sysconfig.get_config_var('PY3LIBRARY')
    if libname1 is None or (isinstance(libname1, str) and len(libname1) == 0):
        libprefix = '.'.join(sysconfig.get_config_var('LIBRARY').split('.')[:-1])
        libname = '.'.join([libprefix, sysconfig.get_config_var('EXT_SUFFIX')])
        libname = libname.replace('..', '.')
        libnames.append(libname)
        if sysconfig.get_config_var('EXT_SUFFIX').find('darwin') != -1:
            libname = '.'.join([libprefix, 'dylib'])
            libnames.append(libname)
    else:
        libnames.append(libname1)
    libname2 = sysconfig.get_config_var('LDLIBRARY')
    if isinstance(libname2, str) and len(libname2) > 0:
        libnames.append(libname2)

    libpath = sysconfig.get_config_var('LIBDIR')
    for libname in libnames:
        pylib = os.path.join(libpath, libname)
        if os.path.exists(pylib):
            return pylib

    libpath2 = os.path.join(libpath, sysconfig.get_config_var('MULTIARCH'))
    for libname in libnames:
        pylib = os.path.join(libpath2, libname)
        if os.path.exists(pylib):
            return pylib

    tail = ''
    prefix = include_dir
    while tail != 'include' and prefix != '/':
        prefix, tail = os.path.split(prefix)
    assert prefix != '/'

    for l in ['lib', 'lib64']:
        libpath = os.path.join(prefix, l)
        for libname in libnames:
            pylib = os.path.join(libpath, libname)
            if os.path.exists(pylib):
                return pylib

        libpath2 = os.path.join(libpath, sysconfig.get_config_var('MULTIARCH'))
        for libname in libnames:
            pylib = os.path.join(libpath2, libname)
            if os.path.exists(pylib):
                return pylib

    assert False


class priism_build(build):
    user_options = [
        ('cxx-compiler=', 'C', 'specify path to C++ compiler'),
        ('python-root-dir=', 'P', 'specify root directory for Python'),
        ('python-include-dir=', 'I', 'specify include directory for Python.h (take priority over python-root-dir)'),
        ('python-library=', 'L', 'specify Python library (take priority over python-root-dir)'),
        ('numpy-include-dir=', 'N', 'specify include directory for NumPy (take priority over python-root-dir)'),
        ('use-intel-compiler=', 'X', 'use intel C++ compiler to build sparseimaging (yes|no)')
    ]

    def initialize_options(self):
        super(priism_build, self).initialize_options()
        self.fftw3_root_dir = None
        initialize_attr_for_user_options(self)
        overwrite_attr_for_user_options_by_environ(self)

    def finalize_options(self):
        super(priism_build, self).finalize_options()
        if self.python_root_dir is None:
            # assuming python executable path to PYTHON_ROOT_DIR/bin/python
            executable_path = sys.executable
            binary_dir, _ = os.path.split(executable_path)
            root_dir, _ = os.path.split(binary_dir)
            self.python_root_dir = root_dir
        if isinstance(self.use_intel_compiler, str) and self.use_intel_compiler.lower() in ('true', 'yes', 'on'):
            self.use_intel_compiler = True
        else:
            self.use_intel_compiler = False
        debug_print_user_options(self)
        print('fftw3-root-dir={}'.format(self.fftw3_root_dir))

    def run(self):
        super(priism_build, self).run()
        for cmd in self.get_sub_commands():
            self.run_command(cmd)

    sub_commands = build.sub_commands + [('build_ext', None)]


class priism_build_ext(build_ext):
    user_options = priism_build.user_options

    def initialize_options(self):
        super(priism_build_ext, self).initialize_options()
        self.fftw3_root_dir = None
        self.priism_build_dir = 'build_ext'
        initialize_attr_for_user_options(self)

    def finalize_options(self):
        super(priism_build_ext, self).finalize_options()
        self.set_undefined_options(
            'build',
            *arg_for_set_undefined_options(self)
        )
        debug_print_user_options(self)

    def run(self):
        super(priism_build_ext, self).run()
        for cmd in self.get_sub_commands():
            self.run_command(cmd)

        self.build_sakura()
        self.build_smili()
        self.install_ext()

    def build_sakura(self):
        execute_command('make sakurapy', cwd=self.priism_build_dir)
        execute_command('cmake -DCOMPONENT=Sakura -P cmake_install.cmake', cwd=self.priism_build_dir)

    def build_smili(self):
        execute_command('make sparseimaging', cwd=self.priism_build_dir)
        execute_command('cmake -DCOMPONENT=Smili -P cmake_install.cmake', cwd=self.priism_build_dir)

    def install_ext(self):
        execute_command('make install/fast', cwd=self.priism_build_dir)

    sub_commands = build_ext.sub_commands + [('configure_ext', None)]


class download_smili(config):
    user_options = []

    def initialize_options(self):
        super(download_smili, self).initialize_options()

        package = 'sparseimaging'
        commit = '4e28903c1fc0256cec5d4b8d5a6371718eff53b9'
        zipname = f'{commit}.zip'
        base_url = f'https://github.com/ikeda46/{package}'
        if IS_GIT_OK:
            url = base_url + '.git'
            def clone_and_checkout():
                execute_command(f'git clone {url}')
                execute_command(f'git checkout {commit}', cwd=package)

            self.download_cmd = clone_and_checkout
        else:
            url = base_url + f'/archive/{zipname}'
            def download_and_extract():
                download_extract(url, filetype='zip')
                os.symlink(f'{package}-{commit}', package)

            self.download_cmd = download_and_extract

        self.package_directory = package

    def finalize_options(self):
        super(download_smili, self).finalize_options()

    def run(self):
        super(download_smili, self).run()

        if not os.path.exists(self.package_directory):
            self.download_cmd()


class download_sakura(config):
    user_options = []

    def initialize_options(self):
        super(download_sakura, self).initialize_options()

        package = 'sakura'
        target = 'libsakura'
        version = 'libsakura-5.1.5'
        zipname = f'{version}.zip'
        base_url = 'https://github.com/tnakazato/sakura'
        if IS_GIT_OK:
            url = base_url + '.git'
            def clone_and_checkout():
                execute_command(f'git clone {url}')
                execute_command(f'git checkout {version}', cwd=package)

            self.download_cmd = clone_and_checkout
        else:
            url = base_url + f'/archive/{zipname}'
            def download_and_extract():
                download_extract(url, filetype='zip')
                os.symlink(f'{package}-{version}', package)

            self.download_cmd = download_and_extract

        self.package_directory = package
        self.target_directory = target

    def finalize_options(self):
        super(download_sakura, self).finalize_options()

    def run(self):
        super(download_sakura, self).run()

        if not os.path.exists(self.package_directory):
            self.download_cmd()

        if not os.path.exists(self.target_directory):
            os.symlink(f'{self.package_directory}/{self.target_directory}', self.target_directory)


class download_eigen(config):
    PACKAGE_NAME = 'eigen'
    PACKAGE_VERSION = '3.3.7'

    user_options = []

    def run(self):
        super(download_eigen, self).run()

        package_directory = f'{self.PACKAGE_NAME}-{self.PACKAGE_VERSION}'
        if not os.path.exists(package_directory):
            tgzname = f'{package_directory}.tar.bz2'
            url = f'https://gitlab.com/libeigen/eigen/-/archive/{self.PACKAGE_VERSION}/{tgzname}'
            download_extract(url, filetype='tar')


class configure_ext(Command):
    user_options = priism_build.user_options

    def initialize_options(self):
        is_cmake_ok = check_command_availability('cmake')
        if not is_cmake_ok:
            raise PriismDependencyError('Command "cmake" is not found. Please install.')
        self.fftw3_root_dir = None
        self.priism_build_dir = None
        self.build_lib = None
        initialize_attr_for_user_options(self)

    def finalize_options(self):
        import numpy

        self.set_undefined_options(
            'build',
            *arg_for_set_undefined_options(self)
        )
        self.set_undefined_options(
            'build_ext',
            ('priism_build_dir', 'priism_build_dir'),
            ('build_lib', 'build_lib')
        )
        if self.python_root_dir is None:
            # assuming python executable path to PYTHON_ROOT_DIR/bin/python
            executable_path = sys.executable
            binary_dir, _ = os.path.split(executable_path)
            root_dir, _ = os.path.split(binary_dir)
            self.python_root_dir = root_dir

        if self.numpy_include_dir is None:
            self.numpy_include_dir = numpy.get_include()

        if self.python_include_dir is None:
            self.python_include_dir = get_python_inc()

        if self.python_library is None:
            self.python_library = get_python_library(self.python_include_dir)

        self.python_version = get_python_version()

        debug_print_user_options(self)
        print('fftw3-root-dir={}'.format(self.fftw3_root_dir))

    def __configure_cmake_command(self):
        cmd = 'cmake -Wno-dev .. -DCMAKE_INSTALL_PREFIX={}'.format(os.path.relpath(self.build_lib, self.priism_build_dir))

        #if self.python_root_dir is not None:
        #    cmd += ' -DPYTHON_ROOTDIR={}'.format(self.python_root_dir)

        cmd += ' -DNUMPY_INCLUDE_DIR={}'.format(self.numpy_include_dir)

        cmd += ' -DPYTHON_INCLUDE_PATH={}'.format(self.python_include_dir)

        cmd += ' -DPYTHON_LIBRARY={}'.format(self.python_library)

        cmd += f' -DPYTHON_VERSION={self.python_version}'

        cmd += f' -DEIGEN_DIR={download_eigen.PACKAGE_NAME}-{download_eigen.PACKAGE_VERSION}'

        cmd += ' -DENABLE_TEST=OFF'

        if self.cxx_compiler is not None:
            cmd += ' -DCMAKE_CXX_COMPILER={}'.format(self.cxx_compiler)


        try:
            if os.environ['USE_INTEL_COMPILER'] in ('true', 'yes', 'on'):
                self.use_intel_compiler = True
            else:
                self.use_intel_compiler = False
        except:
            self.use_intel_compiler = False

        if self.use_intel_compiler is True:
            cmd += ' -DUSE_INTEL_COMPILER=ON'

        #print('generated cmake command:')
        #print('  {}'.format(cmd))
        return cmd

    def run(self):
        # download external packages
        for cmd in self.get_sub_commands():
            self.run_command(cmd)

        # configure with cmake
        if not os.path.exists(self.priism_build_dir):
            os.mkdir(self.priism_build_dir)

        cmd = self.__configure_cmake_command()
        execute_command(cmd, cwd=self.priism_build_dir)

    sub_commands = build_ext.sub_commands + [('download_sakura', None), ('download_smili', None), ('download_eigen', None)]

setup(
    name             = 'priism',
    version          = _get_version(),
    packages         = find_packages('python', exclude=['priism.test']),
    package_dir      = {'': 'python'},
    install_requires = _requires_from_file('requirements.txt'),
    cmdclass={
        'build': priism_build,
        'build_ext': priism_build_ext,
        'download_sakura': download_sakura,
        'download_smili': download_smili,
        'download_eigen': download_eigen,
        'configure_ext': configure_ext,
    },
    # to disable egg compression
    zip_safe = False
)

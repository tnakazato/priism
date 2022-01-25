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
import os
import shlex
import subprocess
import sys
import sysconfig

from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.command.config import config
from distutils.sysconfig import get_python_inc
from setuptools import setup, find_packages, Command


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


PRIISM_VERSION = _get_version()
print('PRIISM Version = {}'.format(PRIISM_VERSION))


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


def execute_command(cmdstring, cwd=None):
    retcode = subprocess.call(shlex.split(cmdstring), cwd=cwd)
    if retcode != 0:
        print('WARNING: command "{}" failed to execute'.format(cmdstring))
    return retcode


def opt2attr(s):
    return s[0].strip('=').replace('-', '_')


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


def get_python_library(include_dir):
    libnames = []
    libname1 = sysconfig.get_config_var('PY3LIBRARY')
    if libname1 is None or (isinstance(libname1, str) and len(libname1) == 0):
        libprefix = '.'.join(sysconfig.get_config_var('LIBRARY').split('.')[:-1])
        libname = '.'.join([libprefix, sysconfig.get_config_var('SO')])
        libname = libname.replace('..', '.')
        libnames.append(libname)
        if sysconfig.get_config_var('SO').find('darwin') != -1:
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

        is_git_ok, is_curl_ok, is_wget_ok = check_command_availability(['git', 'curl', 'wget'])
        package = 'sparseimaging'
        branch = 'smili'
        commit = '77e8d010d2db154aa49280ac6655642929315a49'
        zipname = '{}.zip'.format(commit)
        base_url = 'https://github.com/ikeda46/{}'.format(package)
        if is_git_ok:
            url = base_url + '.git'
            self.download_cmd = 'git clone {}'.format(url)
        elif is_curl_ok:
            url = base_url + '/archive/{}.zip'.format(commit)
            self.download_cmd = 'curl -L -O {}'.format(url)
        elif is_wget_ok:
            url = base_url + '/archive/{}.zip'.format(commit)
            self.download_cmd = 'wget {}'.format(url)
        else:
            raise PriismDependencyError('No download command found: you have to install git or curl or wget')

        if is_git_ok:
            self.epilogue_cmds = ['git checkout {}'.format(commit)]
            self.epilogue_cwd = package
        else:
            self.epilogue_cmds = ['unzip {}'.format(zipname),
                                  'ln -s {0}-{1} {0}'.format(package, commit)]
            self.epilogue_cwd = '.'
        self.package_directory = package

    def finalize_options(self):
        super(download_smili, self).finalize_options()

    def run(self):
        super(download_smili, self).run()

        if not os.path.exists(self.package_directory):
            execute_command(self.download_cmd)
            for cmd in self.epilogue_cmds:
                execute_command(cmd, cwd=self.epilogue_cwd)


class download_sakura(config):
    user_options = []

    def initialize_options(self):
        super(download_sakura, self).initialize_options()

        is_curl_ok, is_wget_ok = check_command_availability(['curl', 'wget'])
        package = 'libsakura'
        version = '5.0.8'
        tgzname = '{}-{}.tgz'.format(package, version)
        url = 'https://alma-intweb.mtk.nao.ac.jp/~sakura/libsakura/{}'.format(tgzname)
        if is_curl_ok:
            self.download_cmd = 'curl -L -O {}'.format(url)
        elif is_wget_ok:
            self.download_cmd = 'wget {}'.format(url)
        else:
            raise PriismDependencyError('No download command found: you have to install curl or wget')

        self.epilogue_cmds = ['tar zxf {}'.format(tgzname)]
        self.epilogue_cwd = '.'
        self.distfile = tgzname
        self.package_directory = package
        self.working_directory = self.package_directory

    def finalize_options(self):
        super(download_sakura, self).finalize_options()

    def run(self):
        super(download_sakura, self).run()

        if not os.path.exists(self.package_directory):
            if not os.path.exists(self.distfile):
                execute_command(self.download_cmd)
            for cmd in self.epilogue_cmds:
                execute_command(cmd, cwd=self.epilogue_cwd)


class download_eigen(config):
    user_options = []

    def initialize_options(self):
        super(download_eigen, self).initialize_options()

        is_curl_ok, is_wget_ok = check_command_availability(['curl', 'wget'])
        package = 'eigen'
        version = '3.3.7'
        tgzname = '{}-{}.tar.bz2'.format(package, version)
        url = 'https://gitlab.com/libeigen/eigen/-/archive/{}/{}'.format(version, tgzname)
        if is_curl_ok:
            self.download_cmd = 'curl -L -O {}'.format(url)
        elif is_wget_ok:
            self.download_cmd = 'wget {}'.format(url)
        else:
            raise PriismDependencyError('No download command found: you have to install curl or wget')

        self.epilogue_cmds = ['tar jxf {}'.format(tgzname)]
        self.epilogue_cwd = '.'
        self.distfile = tgzname
        self.package_directory = f'{package}-{version}'
        self.working_directory = self.package_directory

    def finalize_options(self):
        super(download_eigen, self).finalize_options()

    def run(self):
        super(download_eigen, self).run()

        if not os.path.exists(self.package_directory):
            if not os.path.exists(self.distfile):
                print('Extracting eigen...')
                execute_command(self.download_cmd)
            for cmd in self.epilogue_cmds:
                execute_command(cmd, cwd=self.epilogue_cwd)


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

        debug_print_user_options(self)
        print('fftw3-root-dir={}'.format(self.fftw3_root_dir))

    def __configure_cmake_command(self):
        cmd = 'cmake .. -DCMAKE_INSTALL_PREFIX={}'.format(os.path.relpath(self.build_lib, self.priism_build_dir))

        #if self.python_root_dir is not None:
        #    cmd += ' -DPYTHON_ROOTDIR={}'.format(self.python_root_dir)

        cmd += ' -DNUMPY_INCLUDE_DIR={}'.format(self.numpy_include_dir)

        cmd += ' -DPYTHON_INCLUDE_PATH={}'.format(self.python_include_dir)

        cmd += ' -DPYTHON_LIBRARY={}'.format(self.python_library)

        if self.cxx_compiler is not None:
            cmd += ' -DCMAKE_CXX_COMPILER={}'.format(self.cxx_compiler)

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

        #cache_file = os.path.join(self.priism_build_dir, 'CMakeCache.txt')
        #if os.path.exists(cache_file):
        #    os.remove(cache_file)

        cmd = self.__configure_cmake_command()
        execute_command(cmd, cwd=self.priism_build_dir)

    sub_commands = build_ext.sub_commands + [('download_sakura', None), ('download_smili', None), ('download_eigen', None)]


setup(
    name='priism',
    version=PRIISM_VERSION,
    packages=find_packages('python', exclude=['priism.test']),
    package_dir={'': 'python'},
    install_requires=['numpy'],
    setup_requires=['numpy'],
    cmdclass={
        'build': priism_build,
        'build_ext': priism_build_ext,
        'download_sakura': download_sakura,
        'download_smili': download_smili,
        'download_eigen': download_eigen,
        'configure_ext': configure_ext,
    },
    # to disable egg compression
    zip_safe=False
)

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
import re
import shlex
import ssl
import subprocess
import tarfile
import urllib.request as request
import zipfile

from distutils.command.build_ext import build_ext
from distutils.command.config import config
from distutils.core import Extension
from setuptools import setup, find_packages


def require_from_file(requirement_file):
    with open(requirement_file, 'r') as f:
        requirements = map(
            lambda x: x.rstrip('\n').strip(),
            filter(
                lambda x: len(x.strip()) > 0 and not x.strip().startswith('#'),
                f
            )
        )
        requirements = map(
            lambda x: re.sub(r'^([^=<>]*)([<>=!][=]?)([\d.]*)', r'\1 (\2\3)', x),
            requirements
        )
        requirements = list(requirements)
        print(requirements)
        return requirements


def execute_command(cmdstring, cwd=None):
    retcode = subprocess.call(shlex.split(cmdstring), cwd=cwd)
    if retcode != 0:
        print('WARNING: command "{}" failed to execute'.format(cmdstring))
    return retcode


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


def check_command_availability(cmd):
    if isinstance(cmd, list):
        return [check_command_availability(_cmd) for _cmd in cmd]
    else:
        assert isinstance(cmd, str)
        return subprocess.call(['which', cmd], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) == 0


IS_GIT_OK = check_command_availability('git')


def download_extract(url, filetype):
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    req = request.urlopen(url, context=ssl_context)
    bstream = io.BytesIO(req.read())
    if filetype == 'zip':
        with zipfile.ZipFile(bstream, mode='r') as zf:
            zf.extractall()
    elif filetype == 'tar':
        with tarfile.open(mode='r', fileobj=bstream) as tf:
            tf.extractall()


class download_smili(config):
    user_options = []

    def initialize_options(self):
        super(download_smili, self).initialize_options()

        package = 'sparseimaging'
        commit = '46268c1c66be33a8b09c2ebe4f59a841e3d3b21e'
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


class download_eigen(config):
    PACKAGE_NAME = 'eigen'
    PACKAGE_VERSION = '3.3.7'
    PACKAGE_COMMIT_HASH = '21ae2afd4edaa1b69782c67a54182d34efe43f9c'

    user_options = []

    def run(self):
        super(download_eigen, self).run()

        package_directory = f'{self.PACKAGE_NAME}-{self.PACKAGE_VERSION}'
        if not os.path.exists(package_directory):
            tgzname = f'{package_directory}.tar.gz'
            url = f'https://gitlab.com/libeigen/eigen/-/archive/{self.PACKAGE_VERSION}/{tgzname}'
            download_extract(url, filetype='tar')

            # sometimes directory name is suffixed with commit hash
            if os.path.exists(f'{self.PACKAGE_NAME}-{self.PACKAGE_COMMIT_HASH}'):
                os.symlink(f'{self.PACKAGE_NAME}-{self.PACKAGE_COMMIT_HASH}', package_directory)

        # abort if eigen directory doesn't exist
        if not os.path.exists(package_directory):
            raise FileNotFoundError(f'Failed to download/extract {package_directory}')


project_dir = os.path.dirname(os.path.abspath(__file__))
smili_src_dir = os.path.join(project_dir, 'sparseimaging/c++')


def use_intel_compiler() -> bool:
    return os.environ.get('USE_INTEL_COMPILER', 'no') in ('true', 'yes', 'on')


def get_smili_build_options():
    makefile = os.path.join(smili_src_dir, 'makefile')
    with open(makefile, 'r') as f:
        makefile_contents = f.readlines()

    eigen_include_dir = os.path.join(project_dir, f'{download_eigen.PACKAGE_NAME}-{download_eigen.PACKAGE_VERSION}')
    optimization_flag = '-O3'
    fftw3_include_dir = '/usr/include'
    fftw3_library_dir = '/usr/lib'
    fftw3_libraries = ['fftw3', 'fftw3_omp']
    icpc_flag = '-ipo -qopenmp -xHost'
    cpp_std = '-std=c++11'

    include_dir = [eigen_include_dir, fftw3_include_dir]
    library_dir = [fftw3_library_dir]
    libraries = fftw3_libraries
    extra_compiler_flag = ['-fPIC', cpp_std, optimization_flag]
    if use_intel_compiler():
        extra_compiler_flag.extend(icpc_flag.split())
    extra_link_flag = ['-shared', '-Xlinker', '-rpath', '-Xlinker', '/usr/lib']
    return include_dir, library_dir, libraries, extra_compiler_flag, extra_link_flag


class build_smili(build_ext):
    def build_extensions(self) -> None:
        if use_intel_compiler():
            exe = 'icpc'
        else:
            exe = 'g++'

        self.compiler.set_executable('compiler_so', exe)
        self.compiler.set_executable('compiler_cxx', exe)
        self.compiler.set_executable('linker_so', exe)

        super().build_extensions()


smili_build_options = get_smili_build_options()


requirements = require_from_file('requirements.txt')

setup(
    name='priism',
    version=_get_version(),
    packages=find_packages('python', exclude=['priism.test']),
    package_dir={'': 'python'},
    install_requires=requirements,
    ext_modules=[
        Extension(
            'priism.core.libmfista_fft',
            sources=[
                os.path.join(smili_src_dir, 'mfista_tools.cpp'),
                os.path.join(smili_src_dir, 'mfista_memory.cpp'),
                os.path.join(smili_src_dir, 'mfista_fft_lib.cpp')
            ],
            include_dirs=smili_build_options[0],
            library_dirs=smili_build_options[1],
            libraries=smili_build_options[2],
            extra_compile_args=smili_build_options[3],
            extra_link_args=smili_build_options[4]
        ),
        Extension(
            'priism.core.libmfista_nufft',
            sources=[
                os.path.join(smili_src_dir, 'mfista_tools.cpp'),
                os.path.join(smili_src_dir, 'mfista_memory.cpp'),
                os.path.join(smili_src_dir, 'mfista_nufft_lib.cpp')
            ],
            include_dirs=smili_build_options[0],
            library_dirs=smili_build_options[1],
            libraries=smili_build_options[2],
            extra_compile_args=smili_build_options[3],
            extra_link_args=smili_build_options[4]
        )
    ],
    cmdclass={
        'build_ext': build_smili,
        'download_smili': download_smili,
        'download_eigen': download_eigen,
    },
    # to disable egg compression
    zip_safe=False
)

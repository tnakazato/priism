import os
import shlex
import subprocess

from distutils.command.build_ext import build_ext
from distutils.command.build_clib import build_clib
from distutils.command.config import config
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

version = _get_version()
print('PRIISM Version = {}'.format(version))

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


class build_smili(build_clib):
    user_options = []

    def initialize_options(self):
        super(build_smili, self).initialize_options()
        print('BuildExt.initialize_options')

    def finalize_options(self):
        print('BuildExt.finalize_options')

    def run(self):
        print('BuildExt.run')

class build_sakura(build_ext):
    user_options = []

    def initialize_options(self):
        super(build_sakura, self).initialize_options()
        print('BuildExt.initialize_options')

    def finalize_options(self):
        print('BuildExt.finalize_options')

    def run(self):
        print('BuildExt.run')

class build_priism_ext(build_ext):
    def run(self):
        super(build_priism_ext, self).run()
        for cmd in self.get_sub_commands():
            self.run_command(cmd)
    sub_commands = build_ext.sub_commands + [('build_sakura', None), ('build_smili', None)]


class download_smili(config):
    user_options = []

    def initialize_options(self):
        super(download_smili, self).initialize_options()

        is_git_ok, is_curl_ok, is_wget_ok = check_command_availability(['git', 'curl', 'wget'])
        package = 'sparseimaging'
        branch = 'smili'
        zipname = '{}.zip'.format(branch)
        base_url = 'https://github.com/ikeda46/{}'.format(package)
        if is_git_ok:
            url = base_url + '.git'
            self.download_cmd = 'git clone {}'.format(url)
        elif is_curl_ok:
            url = base_url + '/archive/{}'.format(zipname)
            self.download_cmd = 'curl -L -O {}'.format(url)
        elif is_wget_ok:
            url = base_url + '/archive/{}'.format(zipname)
            self.download_cmd = 'wget {}'.format(url)
        else:
            raise FileNotFoundError('No download command found: you have to install git or curl or wget')

        if is_git_ok:
            self.epilogue_cmds = ['git checkout {}'.format(branch)]
            self.epilogue_cwd = package
        else:
            self.epilogue_cmds = ['unzip {}'.format(zipname),
                                  'ln -s {0}-{1} {0}'.format(package, branch)]
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
        version = '5.0.7'
        tgzname = '{}-{}.tgz'.format(package, version)
        url = 'https://alma-intweb.mtk.nao.ac.jp/~nakazato/libsakura/{}'.format(tgzname)
        if is_curl_ok:
            self.download_cmd = 'curl -L -O {}'.format(url)
        elif is_wget_ok:
            self.download_cmd = 'wget {}'.format(url)
        else:
            raise FileNotFoundError('No download command found: you have to install curl or wget')

        self.epilogue_cmds = ['tar zxvf {}'.format(tgzname)]
        self.epilogut_cwd = '.'
        self.package_directory = package
        self.working_directory = self.package_directory

    def finalize_options(self):
        super(download_sakura, self).finalize_options()

    def run(self):
        super(download_sakura, self).run()

        if not os.path.exists(self.package_directory):
            execute_command(self.download_cmd)
            for cmd in self.epilogue_cmds:
                execute_command(cmd, cwd=self.epilogut_cwd)


class configure_ext(Command):
    user_options = [('eigen3-include-dir=', 'E', 'specify directory for Eigen3'),
                    ('fftw3-root-dir=', 'F', 'specigy root directory for FFTW3'),
                    ('openblas-library-dir=', 'B', 'specify directory for OpenBLAS')]

    def initialize_options(self):
        is_cmake_ok = check_command_availability('cmake')
        if not is_cmake_ok:
            raise FileNotFoundError('Command "cmake" is not found. Please install.')
        self.eigen3_include_dir = None
        self.fftw3_root_dir = None
        self.openblas_libraray_dir = None

    def finalize_options(self):
        print('eigen3-include-dir={}'.format(self.eigen3_include_dir))
        print('fftw3-root-dir={}'.format(self.eigen3_include_dir))
        print('openblas-library-dir={}'.format(self.eigen3_include_dir))

    def run(self):
        # download external packages
        for cmd in self.get_sub_commands():
            self.run_command(cmd)

        # configure with cmake

    sub_commands = build_ext.sub_commands + [('download_sakura', None), ('download_smili', None)]



setup(
   name='priism',
   version=version,
   packages=find_packages('python', exclude=['priism.test']),
   package_dir={'':'python'},
   install_requires=['numpy'],
   cmdclass={'build_sakura': build_sakura, 
             'build_smili': build_smili,
             'build_ext': build_priism_ext,
             'download_sakura': download_sakura,
             'download_smili': download_smili,
             'configure_ext': configure_ext}
)
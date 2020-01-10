import os
import shlex
import subprocess

from distutils.command.build_ext import build_ext
from distutils.command.build_clib import build_clib
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


def check_command_availability(cmdlist):
    return [subprocess.call(['which', cmd], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) == 0 for cmd in cmdlist]


def execute_command(cmdstring, cwd=None):
    retcode = subprocess.call(shlex.split(cmdstring), cwd=cwd)
    if retcode != 0:
        print('WARNING: command "{}" failed to execute'.format(cmdstring))
    return retcode


class configure_smili(Command):
    user_options = []

    def initialize_options(self):
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
            self.epilogut_cwd = package
        else:
            self.epilogue_cmds = ['unzip {}'.format(zipname),
                                  'ln -s {0}-{1} {0}'.format(package, branch)]
            self.epilogut_cwd = '.'
        self.directory = package

    def finalize_options(self):
        pass

    def run(self):
        if not os.path.exists(self.directory):
            execute_command(self.download_cmd)
            for cmd in self.epilogue_cmds:
                execute_command(cmd, cwd=self.epilogut_cwd)


class configure_sakura(Command):
    user_options = []

    def initialize_options(self):
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
        self.directory = package

    def finalize_options(self):
        pass

    def run(self):
        if not os.path.exists(self.directory):
            execute_command(self.download_cmd)
            for cmd in self.epilogue_cmds:
                execute_command(cmd, cwd=self.epilogut_cwd)


class configure_ext(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for cmd in self.get_sub_commands():
            self.run_command(cmd)
    sub_commands = build_ext.sub_commands + [('configure_sakura', None), ('configure_smili', None)]



setup(
   name='priism',
   version=version,
   packages=find_packages('python', exclude=['priism.test']),
   package_dir={'':'python'},
   install_requires=['numpy'],
   cmdclass={'build_sakura': build_sakura, 
             'build_smili': build_smili,
             'build_ext': build_priism_ext,
             'configure_sakura': configure_sakura,
             'configure_smili': configure_smili,
             'configure_ext': configure_ext}
)
from __future__ import print_function, division
import distutils.sysconfig
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.install_data import install_data
import numpy as np
import os, sys

# Must have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a ")
    print("copy from www.cython.org and install it")
    sys.exit(1)

# Scan directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

# Generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [np.get_include(), "."],   # the '.' is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall"],
        extra_link_args = [],
        libraries = [],
        )


# Setup Variables
dirname = 'metpy'
packages = ['metpy']
setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, dirname))
module_path = distutils.sysconfig.get_python_lib()
include_files = ['LICENSE', 'README.md']
data_files = [(os.path.join(module_path, dirname), include_files)]

import version
version.write_git_version()
ver = version.get_version()
sys.path.pop()

# Get the list of extensions
extNames = scandir(dirname)

# Build the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

setup(
    name                    = 'MetPy',
    version                 = ver,
    description             = 'Meteorological Python',
    author                  = 'Patrick Marsh, Ryan May',
    author_email            = 'pmarshwx@gmail.com, rmay31@gmail.com',
    url                     = '',
    download_url            = '',
    packages                = packages,
    data_files              = data_files,
    scripts                 = [],
    cmdclass                = {'build_ext': build_ext},
    ext_modules             = extensions,   )


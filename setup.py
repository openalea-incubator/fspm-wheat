# -*- coding: latin-1 -*-

import ez_setup

import sys
from setuptools import setup, find_packages

import fspmwheat

"""
Notes:

- use setup.py develop when tracking in-development code
- when removing modules or data files from the project, run setup.py clean --all and delete any obsolete .pyc or .pyo.

"""

ez_setup.use_setuptools()

if sys.version_info < (2, 7):
    print('ERROR: FSPM-Wheat requires at least Python 2.7 to run.')
    sys.exit(1)

if sys.version_info >= (3, 0):
    print('WARNING: FSPM-Wheat has not been tested with Python 3.')

setup(
    name="FSPM-Wheat",
    version=fspmwheat.__version__,
    packages=find_packages(),

    # install_requires=['numpy>=1.7.2', 'pandas>=0.14.0', 'scipy>=0.12.1', 'matplotlib>=1.3.1'],
    include_package_data=True,

    # metadata for upload to PyPI
    author="C.Chambon, M.Gauthier, R.Barillot",
    author_email="camille.chambon@inra.fr, romain.barillot@inra.fr",
    description="A Functional Structural Plant Model of Wheat",
    long_description="An example of model coupling for building a Functional Structural Plant Model of Wheat",
    license="CeCILL-C",
    keywords="functional-structural plant model, wheat, metabolism, resource acquisition and allocation, model coupling, OpenAlea, MTG",
    url="https://sourcesup.renater.fr/projects/fspm-wheat/",
    download_url="",  # TODO
)

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import requests
import zipfile
import atexit
from distutils.command.install import INSTALL_SCHEMES

setup(
    name='atlaspy',
    version='0.0.15',
    description='Python library for working with brain atlases',
    author='Alfredo Lucas',
    author_email='alfredo1238@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'atlaspy.source_data': ["*"]},

    install_requires=['pyvista', 'seaborn', 'pandas', 'matplotlib', 'nibabel', 'nilearn', 'trame','ipywidgets'],

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
    ],
)

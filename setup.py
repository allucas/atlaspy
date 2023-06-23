from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import requests
import zipfile
import atexit
from distutils.command.install import INSTALL_SCHEMES

def download_files():
    # Define the URL from which to download the files

    # v1: zip_url = 'http://dl.dropboxusercontent.com/scl/fi/s88fszf6t1q6ef4znl6cr/stls.zip?dl=0&rlkey=j93ehij42d3g0rp1hqn9u0f5t'
    zip_url = 'http://dl.dropboxusercontent.com/scl/fi/a7usav2cmyyskdb339pzu/stls_v2.zip?dl=0&rlkey=3zxd59bginwkkvnte1ojrzfk2'

    # Create the target directory if it doesn't exist
    stl_dir = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'atlaspy/source_data/atlases/')
    os.makedirs(stl_dir, exist_ok=True)

    # Download the ZIP file
    zip_path = os.path.join(stl_dir, 'stls.zip')
    response = requests.get(zip_url)
    with open(zip_path, 'wb') as file:
        file.write(response.content)

    # Extract the STL files from the ZIP file preserving folder structure
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(stl_dir)

    # Remove the downloaded ZIP file
    os.remove(zip_path)

# Custom installation command that runs the download_files function after installation
class PostInstallCommand(install):
    def __init__(self, *args, **kwargs):
        print('Downloading required files')
        super().__init__(*args, **kwargs)
        atexit.register(download_files)

# Modify the data install dir to match the source install dir
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']


setup(
    name='atlaspy',
    version='0.0.13',
    description='Python library for working with brain atlases',
    author='Alfredo Lucas',
    author_email='alfredo1238@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={'atlaspy.source_data': ["*"]},

    install_requires=['pyvista', 'seaborn', 'pandas', 'matplotlib', 'nibabel', 'nilearn'],
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
    ],
)

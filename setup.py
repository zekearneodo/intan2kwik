from setuptools import setup
import sys

if sys.version_info < (2, 7):
    sys.exit('Sorry, Python < 2.7 is not supported')

setup(name='intan2kwik',
      version='0.0.2',
      description='Tools to convert intan to Kwik format',
      url='http://github.com/zekearneodo/intan2kwik',
      author='Zeke Arneodo',
      author_email='ezequiel@ini.ethz.ch',
      license='GNU3',
      packages=['intan2kwik'],
      install_requires=['numpy>=1.4',
                        'h5py>=2.9',
                        'tqdm>=4.28',
                        'numba>=0.4',
                        'pandas>=0.23'],
      zip_safe=False)

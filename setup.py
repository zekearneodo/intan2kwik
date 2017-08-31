from setuptools import setup
import sys

if sys.version_info < (2, 7):
    sys.exit('Sorry, Python < 2.7 is not supported')

setup(name='intan2kwik',
      version='0.0.1',
      description='Tools to convert intan to Kwik format',
      url='http://github.com/zekearneodo/intan2kwik',
      author='Zeke Arneodo',
      author_email='earneodo@ucsd.edu',
      license='GNU3',
      packages=['intan2kwik'],
      install_requires=[],
      zip_safe=False)

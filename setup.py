import os
import sys
from setuptools import setup, find_packages

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    install_requires = []
    ext_modules = []
    include_dirs = []
    cython_exts = []
else:
    import numpy as np

    # ---- C/C++ EXTENSIONS ---- #
    include_dirs = [np.get_include()]

    install_requires = ['menpo>=0.3.0',
                        'scikit-image>=0.10.1',
                        'menpofit']

setup(name='cvpr15',
      version='0.0',
      description='CVPR 2015',
      author='Epameinondas Antonakos',
      author_email='antonakosn@gmail.com',
      include_dirs=include_dirs,
      packages=find_packages(),
      install_requires=install_requires
)

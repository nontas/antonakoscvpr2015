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
    from Cython.Build import cythonize
    import numpy as np

    # ---- C/C++ EXTENSIONS ---- #
    cython_modules = ["cvpr15/cython/compute_gradient.pyx",
                      "cvpr15/cython/extract_patches.pyx"]

    cython_exts = cythonize(cython_modules, quiet=True)
    include_dirs = [np.get_include()]

    install_requires = ['menpo>=0.3.0',
                        'scikit-image>=0.10.1']

setup(name='cvpr15',
      version='0.0',
      description='CVPR 2015',
      author='Epameinondas Antonakos',
      author_email='antonakosn@gmail.com',
      include_dirs=include_dirs,
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=install_requires
)

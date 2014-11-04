from setuptools import setup, find_packages
import numpy as np

include_dirs = [np.get_include()]

requirements = ['menpo>=0.3.0',
                'menpofast>=0.0.1',
                'menpofit>=0.0.1',
                'scikit-image>=0.10.1']

setup(name='antonakoscvpr15',
      version='0.0.1',
      description='CVPR 2015',
      author='Epameinondas Antonakos',
      author_email='antonakosn@gmail.com',
      include_dirs=include_dirs,
      packages=find_packages(),
      install_requires=requirements
)

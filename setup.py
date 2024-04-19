from setuptools import setup, find_packages

setup(
    name='HSI-Cls',
    description='Hyperspectral image classification',
    version='0.1',
    packages=find_packages(include=['hsicls']),
    install_requires=[
        'spectral',
        'scipy',
        'numpy',
        'scikit-learn',
        'tabulate',
        'hdf5storage',
    ],
)

#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-model-specs',
    version='1.2.0',
    description='A helper package for managing Keras model base architectures with overrides for target size and preprocessing functions.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    include_package_data=True,
    install_requires=[
        'Keras',
        'Keras-Applications>=1.0.7',
        'h5py',
        'Pillow',
        'six'
    ]
)

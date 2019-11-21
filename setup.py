#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='keras-model-specs',
    version='2.0.1',
    description='A helper package for managing tf.keras model base architectures with overrides for target size '
                'and preprocessing functions.',
    author='Triage Technologies Inc.',
    author_email='ai@triage.com',
    url='https://www.triage.com/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    include_package_data=True,
    install_requires=[
        'tensorflow>=2.0',
        'h5py',
        'Pillow',
        'six'
    ]
)

# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='CoffeeLibs',
    version='1.0',
    description='PyCoffee: COFFEE  Coronagraphic Focal plane wave-Front Estimation for Exoplanet detection',
    packages=['CoffeeLibs'],
    author='Sandrine Juillard',
    author_email='sandrine.juillard@hotmail.fr',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        ],
    keywords='COFFEE',
    install_requires=['Asterix','numpy', 'scipy','matplotlib', 'configobj']
    )


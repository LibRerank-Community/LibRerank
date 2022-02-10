#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
from setuptools import setup, find_packages

DESCRIPTION = ''' librerank is a toolkit for re-ranking algorithms. There are a number of re-ranking algorithms, such as PRM, DLCM, GSF, miDNN, SetRank, EGRerank, Seq2Slate. It support SVMRank, LambdaMART, and DNN as initial ranker. '''

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name="librerank",
    version="0.1.0",
    author="Yunjia Xi & Weiwen Liu",
    author_email="yunjiaxi404@gmail.com & liuweiwen1995@gmail.com",
    description=(DESCRIPTION),
    license="MIT",
    keywords="TODO",
    url="TODO",
    packages=find_packages(),
    long_description=long_description,
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=[
        'tensorflow-gpu >= 1.9.0,<2',
        # 'tensorflow >= 1.9.0,<2',
        'numpy >= 1.16.4',
        'scikit-learn >= 0.21.2',
        'lightgbm >= 2.3.2',
    ]
)
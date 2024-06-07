#!/usr/bin/python3

from setuptools import setup

setup(
    name='qfcerrt_noot', 
    version='0.1',
    author='Noah Otte',
    description='An RRT based 2D pathplanner utilizing quadtrees.',
    long_description=open('README.md').read(),
    author_email='nvethrandil@gmail.com',
    url='https://github.com/Nvethrandil/QFCERRT',
    keywords='development, rrt, pathplanner',
    python_requires='>=3.8',
    packages=['qfcerrt_noot'],
    install_requires=[
        'scipy',
        'scikit-image',
        'numpy',
        'matplotlib',
        'rospy',
        'roslib',
        'numpy-ros',
        'rosmsg'
    ],
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ]
)
#!/usr/bin/env python
import sys
import os
import glob
from torch.utils import cpp_extension
from setuptools import setup, find_packages, Extension
import numpy as np
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from distutils.core import setup, Extension
setup(name='pynvjpeg',
    version='0.0.7',
    #ext_modules=[Extension('nvjpeg', ['nvjpeg-python.c'],
    #                       include_dirs=[np.get_include(),'/usr/local/cuda/include'])],
    ext_modules=[cpp_extension.CppExtension('nvjpeg',['nvjpeg-python.cpp'],
                             include_dirs=[np.get_include(),'/usr/local/cuda/include'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    author="Usingnet",
    author_email="developer@usingnet.com",
    license="MIT",
    description="nvjpeg for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UsingNet/nvjpeg-python",
    # packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA :: 10.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    keywords=[
        "pynvjpeg",
        "nvjpeg",
        "jpeg",
        "jpg",
        "encode",
        "decode",
        "jpg encode",
        "jpg decode",
        "jpeg encode",
        "jpeg decode",
        "gpu",
        "nvidia"
    ],
    python_requires=">=3.6",
    project_urls={
        'Source': 'https://github.com/UsingNet/nvjpeg-python',
        'Tracker': 'https://github.com/UsingNet/nvjpeg-python/issues',
    },
    install_requires=['numpy>=1.17']
)

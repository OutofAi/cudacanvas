from setuptools import setup
import os
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

cwd = os.path.dirname(os.path.abspath(__file__))
include_path = None
lib_path = None

with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

if sys.platform.startswith('linux'):
    glfw_libraries = ['glfw', 'GL']
elif sys.platform.startswith('win'):
    glfw_libraries = ['glfw3dll', 'opengl32']
else:
    raise Exception("Unsupported platform")


if 'INCLUDE_PATH' in os.environ:
    include_path = os.environ['INCLUDE_PATH']
else:
    raise Exception("You need to provide a INCLUDE_PATH for GLFW library")

if 'LIB_PATH' in os.environ:
    lib_path = os.environ['LIB_PATH']
else:
    raise Exception("You need to provide a LIB_PATH for GLFW library")

ext_modules = [
    CUDAExtension(
        'cudacanvas.cudaGLStream',
        ['cudacanvas/cudacanvas.cpp'], 
        include_dirs=[include_path],
        library_dirs=[lib_path],
        libraries=glfw_libraries,
        language='c++'
    ),
]

cuda_version = torch.version.cuda.replace(".", "")

# Fetching the CUDA version from PyTorch and formatting the version string
torch_version = torch.__version__.split('+')[0]  # Gets the base version of torch, e.g., '2.2.2'
cuda_version = torch.version.cuda.replace(".", "")  # Gets CUDA version, e.g., '118'
version_base = torch_version  # Base version now uses the PyTorch version

full_version = version_base

setup(
    name='cudacanvas',
    version=full_version,
    author='Ashleigh Watson & Alex Nasa',
    url='https://github.com/OutofAi/cudacanvas',
    packages=['cudacanvas'],
    description='Real-time PyTorch Tensor Visualisation in CUDA, Eliminating CPU Transfer',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            'License :: OSI Approved :: MIT License',
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'glfw'
    ]
)

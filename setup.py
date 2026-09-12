from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "gradiex",
        ["python/gradiex_module.cpp"],
        include_dirs=["."],          # so the binding can #include "core/network.h"
        cxx_std=20,                  # core/ uses C++20 features
        extra_compile_args=["-O2", "-Wno-unknown-pragmas"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

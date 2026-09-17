from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# The extension is gradiex._core; gradiex/__init__.py re-exports it and adds the
# matplotlib-backed plotting layer on top.
ext_modules = [
    Pybind11Extension(
        "gradiex._core",
        ["python/gradiex_module.cpp"],
        # Without `depends`, setuptools only times-stamps the .cpp: editing a
        # header leaves a stale .so in place and the next run silently tests the
        # previous engine.
        depends=sorted(glob("core/*.h")),
        include_dirs=["."],          # so the binding can #include "core/network.h"
        cxx_std=20,                  # core/ uses C++20 features
        extra_compile_args=["-O2", "-Wno-unknown-pragmas", "-Wno-ignored-qualifiers"],
    ),
]

setup(packages=["gradiex"], ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

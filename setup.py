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
        # previous engine. It also puts core/*.h into the sdist.
        depends=sorted(glob("core/*.h")),
        include_dirs=["."],          # so the binding can #include "core/network.h"
        cxx_std=20,                  # core/ uses C++20 features
    ),
]


class BuildExt(build_ext):
    """Attach optimisation and warning flags in the spelling each compiler knows.

    core/ carries `#pragma omp simd` throughout, which every compiler warns about
    when OpenMP is off -- the default here, since the pragmas are a hint and the
    engine is correct without them. Each toolchain spells that suppression
    differently, and MSVC does not understand the GCC/Clang `-Wno-` forms at all.
    """

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            args = [
                "/O2",
                "/wd4068",  # unknown pragma -- `omp simd` without /openmp
            ]
        else:
            args = [
                "-O2",
                "-Wno-unknown-pragmas",
                "-Wno-ignored-qualifiers",
            ]
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args) + args
        super().build_extensions()


setup(packages=["gradiex"], ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})

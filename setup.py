from setuptools import setup, Extension
import pybind11

ext = Extension(
    name="profile_hmm",
    sources=["profile_hmm_bindings.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++17", "-O2"],
)

setup(
    name="profile_hmm",
    version="0.1.0",
    ext_modules=[ext],
)

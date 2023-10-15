import os, sys
from setuptools import Extension, setup, find_packages
from typing import List


def get_libnames(libname: str, root_dir: str) -> List[str]:
  opencv_libs = [
    f.split(".")[0][3:] for f in os.listdir(root_dir) if libname in f
  ]
  return opencv_libs


if __name__ == "__main__":
  # Construct include directories.
  usr_include = "/usr/include/"
  usr_local_include = "/usr/local/include/"
  include_libs = ["", "eigen3", "absl", "opencv4", "ceres", "yaml-cpp"]

  include_dirs = []
  for lib in include_libs:
    for lib_dir in [usr_include, usr_local_include]:
      include_dirs += [os.path.join(lib_dir, lib) for lib in include_libs]
  include_dirs += [
    ".", "calico/third_party/apriltags",
  ]

  # Construct library directories and linker flags.
  usr_lib_dir = "/usr/lib/x86_64-linux-gnu"
  usr_local_lib_dir = "/usr/local/lib"
  library_dirs = [
    usr_lib_dir, usr_local_lib_dir
  ]
  libs = ["absl", "opencv", "ceres", "yaml-cpp"]
  lib_names = ["stdc++"]
  for lib_dir in library_dirs:
    for lib in libs:
      lib_names += get_libnames(lib, lib_dir)

  print(sorted(lib_names))

  # Construct source files.
  src_files = list()
  for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
      if ((name.endswith(".cpp") or name.endswith(".cc")) and "test" not in name):
        src_files.append(os.path.join(root, name))

  # Construct pybind11 extensions.
  from pybind11.setup_helpers import Pybind11Extension, build_ext
  extensions = [
    Pybind11Extension(
      "calico",
      src_files,
      language="c++",
      include_dirs=include_dirs,
      library_dirs=library_dirs,
      runtime_library_dirs=library_dirs,
      libraries=lib_names,
      cxx_std="17",
      extra_compile_args=["-fPIC", "-O3"]
    ),
  ]

  setup(
    name="calico",
    url="https://github.com/yangjames/Calico.git",
    packages=find_packages(where="calico"),
    package_dir={"": "calico"},
    ext_modules=extensions,
    package_data={
      "": ["README.md", "LICENSE"]
    },
    include_package_data=True,
    cmdclass=dict(build_ext=build_ext),
  )

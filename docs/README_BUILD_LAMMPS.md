# Building LAMMPS-METATENSOR from source with KOKKOS support

The original LAMMPS-METATENSOR package is distributed via Anaconda,
where the pre-build binaries are available for CPU and GPU versions
of the code. This allows for easy installation, which does not reqiure
building from source, and thus make the life easier for the users.

However, if the conda installation didn't work for you (for example,
if additional LAMMPS packages or compilation flags are required), you
can build the LAMMPS-METATENSOR from source. This guide will help
you to build the KOKKOS-enabled version of LAMMPS-METATENSOR with
PET-MAD support.

## Prerequisites

To build LAMMPS with KOKKOS support we will use the `conda-build` tool. This will
allow us replicat as close as possible the Anaconda build environment, that is used
to build the main LAMMPS-METATENSOR package.. To start, we need first install
[`Miniforge`](https://github.com/conda-forge/miniforge), a minimal conda installer.
To install Miniforge on unix-like systems, run:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Next, we need to create a new environment and install the `conda-build` tool:

```bash
conda create -n lammps-kokkos
conda activate lammps-kokkos
conda install conda-build
```

## Creating the build recipe

Conda-build allow to use a pre-defined recipe for building the package and perform
the build process in the isolated environment. This assures that the build process
is reproducible and the package is built with the same dependencies as the original
LAMMPS-METATENSOR package. The recipe is a directory with the following structure:

```
lammps-kokkos\
  - meta.yaml
  - build.sh
  - conda_build_config.yaml
```

The `meta.yaml` file contains the package metadata, such as the package name, version,
dependencies, and the source code URL. The `build.sh` script is the build script that
controls the build process. The `conda_build_config.yaml` file contains the configuration
options for the build process.

The `meta.yaml` file for the LAMMPS-KOKKOS package is as follows:

```yaml
{% set version = "2Apr2025" %}
{% set date = datetime.datetime.strptime(version, "%d%b%Y") %}
{% set conda_version = "{:%Y.%m.%d}".format(date) %}

{% set build = 1 %}
{% set git_rev = "12b2d26ed286fd4d8c4c7a340257aecb2e4b0e55" %}
# increase this by 1 everytime you change the git commit above without also
# changing the `version`
{% set git_rev_count = "0" %}

# Use a higher build number for prefered variants, to ensure that they are
# picked first by conda's solver, and installed where the platform supports it.
{% if cuda_compiler_version == "None" %}
# pick CPU before GPU
{% set build = build + 10000 %}
{% endif %}

# pick openmpi first, then mpich, then no-mpi variant
{% if mpi == 'openmpi' %}
  {% set build = build + 2000 %}
{% endif %}

{% if mpi == 'mpich' %}
  {% set build = build + 1000 %}
{% endif %}

{% if mpi != 'nompi' %}
  {% set mpi_prefix = "mpi_" + mpi %}
{% else %}
  {% set mpi_prefix = "nompi" %}
{% endif %}

{% if cuda_compiler_version != "None" %}
  {% set cuda_prefix = "cuda" + environ.get("cuda_compiler_version", "0.0").replace(".", "") %}
  {% set cuda_prefix = cuda_prefix + "_kokkos_arch_" + environ.get("kokkos_arch", "") %}
{% else %}
  {% set cuda_prefix = "cpu" %}
{% endif %}


package:
  name: lammps-metatensor
  version: {{ conda_version }}.mts{{ git_rev_count }}

source:
  git_url: https://github.com/metatensor/lammps.git
  git_rev: {{ git_rev }}
  depth: 1
  patches:
    - kokkos-log2-vectorsize.patch


build:
  number: {{ build }}
  skip: True  # [win]
  skip: True  # [cuda_compiler_version == "11.8"]
  skip: True  # [linux and kokkos_arch == "None" and cuda_compiler_version != "None"]
  skip: True  # [linux and kokkos_arch != "None" and cuda_compiler_version == "None"]
  string: {{cuda_prefix}}_h{{ PKG_HASH }}_{{ mpi_prefix }}_git.{{ git_rev[:7] }}_{{ PKG_BUILDNUM }}


requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - {{ stdlib("c") }}
    - {{ compiler('cuda') }}  # [cuda_compiler_version != "None"]
    - {{ mpi }}               # [build_platform != target_platform and mpi == "openmpi"]
    - make
    - cmake
    - pkg-config
    - llvm-openmp  # [osx]
    - libgomp      # [linux]

  host:
    - {{ mpi }}  # [mpi != 'nompi']
    - cuda-version {{ cuda_compiler_version }} # [cuda_compiler_version != "None"]
    - cuda-toolkit  {{ cuda_compiler_version }} # [cuda_compiler_version != "None"]
    - fftw * {{ mpi_prefix }}_*
    - plumed
    - libmetatensor-torch >=0.7.6,<0.8.0
    # always build against the CPU version of libtorch, we can still pick the
    # cuda one at runtime
    - libtorch * cpu*
  run:
    - {{ mpi }}  # [mpi != 'nompi']
    - plumed
    - libmetatensor-torch >=0.7.6,<0.8.0
    - libmetatensor >=0.1.14,<0.2.0
    - {{ pin_compatible('cuda-version', max_pin='x', min_pin='x') }}  # [cuda_compiler_version != "None"]
    - libcufft  # [cuda_compiler_version != "None"]

test:
  commands:
    - export OMPI_MCA_plm_rsh_agent=false
    - lmp -h  # [mpi == 'nompi' and cuda_compiler_version == "None"]
    - mpirun -np 2 lmp -h  # [mpi != 'nompi' and cuda_compiler_version == "None"]

about:
  home: https://docs.metatensor.org/metatomic/latest/engines/lammps.html
  license:  GPL-2.0-only
  license_family: GPL
  summary: 'Metatensor-enabled version of LAMMPS'
  description: Metatensor-enabled version of LAMMPS
  doc_url: https://docs.metatensor.org/metatomic/latest/engines/lammps.html
  dev_url: https://github.com/metatensor/lammps

extra:
  recipe-maintainers:
    - abmazitov
    - Luthaf
```

The `build.sh` script is as follows:

```bash
#!/bin/bash

PLATFORM=$(uname)

if [[ "$PLATFORM" == 'Darwin' ]]; then
  BUILD_OMP=OFF
elif [[ "$PLATFORM" == 'Linux' ]]; then
  BUILD_OMP=ON
  if [[ ${cuda_compiler_version} != "None" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DPKG_KOKKOS=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON"
    # inspired by lammps' `cmake/presets/kokkos-cuda.cmake`
    CMAKE_ARGS="$CMAKE_ARGS -DFFT_KOKKOS=CUFFT"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=$(pwd)/lib/kokkos/bin/nvcc_wrapper"
    CMAKE_ARGS="$CMAKE_ARGS -DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF"
    CMAKE_ARGS="$CMAKE_ARGS -DKokkos_ARCH_${kokkos_arch}=ON"

    # silent a warning about "calling a constexpr __host__ function from a __host__ __device__ function"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS=--expt-relaxed-constexpr"
  else
    # Make sure to link to `libtorch.so` and not just `libtorch_cpu.so`. This
    # way, the code will try to load `libtorch_cuda.so` as well, enabling cuda
    # device where available even when not using kokkos.
    export LDFLAGS="-lm -ldl -Wl,--no-as-needed,$PREFIX/lib/libtorch.so -Wl,--as-needed"
  fi
fi

if [ "${mpi}" == "nompi" ]; then
  ENABLE_MPI=OFF
else
  ENABLE_MPI=TRUE
fi


mkdir build && cd build

cmake -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DLAMMPS_INSTALL_RPATH=ON \
      -DBUILD_OMP=$BUILD_OMP \
      -DENABLE_MPI=$ENABLE_MPI \
      -DWITH_JPEG=OFF \
      -DWITH_PNG=OFF \
      -DPKG_REPLICA=ON \
      -DPKG_MC=ON \
      -DPKG_MOLECULE=ON \
      -DPKG_MISC=ON \
      -DPKG_PLUMED=ON \
      -DDOWNLOAD_PLUMED=OFF \
      -DPLUMED_MODE="shared" \
      -DPKG_KSPACE=ON \
      -DPKG_MANIFOLD=ON \
      -DPKG_ML-METATENSOR=ON \
      -DDOWNLOAD_METATENSOR=OFF \
      -DPKG_QTB=ON \
      -DPKG_REACTION=ON \
      -DPKG_RIGID=ON \
      -DPKG_SHOCK=ON \
      -DPKG_SPIN=ON \
      -DPKG_MPIIO=$ENABLE_MPI \
      -DPKG_EXTRA_PAIR=ON \
      $CMAKE_ARGS \
      ../cmake

cmake --build . --parallel ${CPU_COUNT} -- VERBOSE=1
cmake --build . --target install
```

In this file, it is possible to add or remove the LAMMPS packages, which are
compiled with the LAMMPS. Please be careful while doing this: removing the
packages will most likely not break the build process, but adding the packages
might require either additional dependencies or additional build flags to be
specified.

The `conda_build_config.yaml` file is as follows:

```yaml
c_compiler:
- gcc
c_compiler_version:
- '13'
c_stdlib:
- sysroot
c_stdlib_version:
- '2.17'
cdt_name:
- conda
channel_sources:
- conda-forge
cuda_compiler:
- cuda-nvcc
cuda_compiler_version:
- '12.6'
cxx_compiler:
- gxx
cxx_compiler_version:
- '13'
docker_image:
- quay.io/condaforge/linux-anvil-x86_64:alma9
fftw:
- '3'
gsl:
- '2.7'
mpi:
- openmpi
kokkos_arch:
- AMPERE80
mpich:
- '4'
openmpi:
- '4'
target_platform:
- linux-64
```

Copy-paste the above files into the `lammps-kokkos` directory. The next step is to
build the package.

## Building the package

In order to adapt the build recipe for your system, you need to check that the 
following variables are set correctly in the `conda_build_config.yaml`

- `kokkos_arch` - check that here the correct version of the GPU architecture is set. 
  The available options can be found
    [here](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#nvidia-gpus), 
    and [here](https://developer.nvidia.com/cuda-gpus). 
- `mpi` - check that the correct MPI implementation is set in.
  The avaiable options are: `nompi`, `mpich`, `openmpi`.

The package can be built using the following command (to be run from the "root" directory of this
guide, i.e., the one which contains `lammps-kokkos\`):

```bash
conda build lammps-kokkos --croot lammps-kokkos-build-artifacts --output-folder lammps-kokkos-build 
```

After this, the build process is started, and the package is built. This might take some time.
The `--croot` flag specifies the directory where the build artifacts are stored, and 
`--output-folder` specifies the path for the built `.conda` bundle. Package name will depend
on the version of the LAMMPS, CUDA, Python, and MPI, so mind that the exact name can be
different in your case. To verify that your package was properly built and now available for
the installation, run:

```bash
conda search -c file://$PWD/lammps-kokkos-build lammps-metatensor
```

This will use the local conda repository to search for the package. If the package
is found, you can proceed to the next step.

## Installing the package

To install the built package, run:

```bash
conda install -c file://$PWD/lammps-kokkos-build lammps-metatensor
```

This will install the package into the current conda environment. You might need to
check that the package which is planned to install is the same, as the one you built.
Otherwise, copy the exact name of the package from the output of the `conda search`
command.

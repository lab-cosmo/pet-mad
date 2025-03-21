# Building LAMMPS-METATENSOR with KOKKOS support

The original LAMMPS-METATENSOR package is distributed via Anaconda,
where the pre-build binaries are available for CPU and GPU versions
of the code. This allows for easy installation, which does not reqiure
building from source, and thus make the life easier for the users.

By default, KOKKOS can be only compiled for a specific GPU architecture.
In the Anaconda distributions we therefore compiled the LAMMPS with KOKKOS
package against the lowest supported GPU architecture, which is Keppler (3.5)
in the case of CUDA 11.8 build and Maxwell (5.0) in the case of CUDA version
12.6. However, it is possible to run the code more modern GPU architectures
thanks to PTX, which translates the CUDA code on-the-fly. 

Unfortunately, this pipeline was broken in CUDA version 12.6, which makes
impossible using the code for the modern GPUs. More details are available
in [this](https://matsci.org/t/support-for-multiple-gpu-architectures/61693/8)
thread. Thus, to use KOKKOS-enabled LAMMPS on the modern GPUs, one needs to
build the code from source. This document describes how to do it.

## Prerequisites

To build LAMMPS with KOKKOS support we will use the `conda-build` tool, to
replicated as close as possible the Anaconda build environment. To start,
we need first install [`Miniforge`](https://github.com/conda-forge/miniforge), 
a minimal conda installer. To install Miniforge on unix-like systems, run:

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
{% set version = "27Jun2024" %}
{% set build = 0 %}
{% set cuda_major = environ.get("cuda_compiler_version", "0.0").split(".")[0]|int %}

# Use a higher build number for the CUDA variant, to ensure that it's
# preferred by conda's solver, and it's preferentially
# installed where the platform supports it.
{% if cuda_compiler_version != "None" %}
{% set build = build + 200 %}
{% endif %}

{% if mpi == 'mpich' %}
{% set build = build + 50 %}
{% endif %}

{% if mpi == 'openmpi' %}
{% set build = build + 100 %}
{% endif %}


package:
  name: lammps-metatensor
  version: {{ version }}

source:
  git_url: https://github.com/metatensor/lammps.git
  depth: 1
{% if cuda_compiler_version != "None" %}
  git_rev: kokkos-pet-mad
{% endif %}
build:
  number: {{ build }}
  {% if mpi != 'nompi' %}
  {% set mpi_prefix = "mpi_" + mpi %}
  {% else %}
  {% set mpi_prefix = "nompi" %}
  {% endif %}
  skip: True  # [win]
  skip: True  # [cuda_compiler_version == "10.2"]
  skip: True  # [cuda_compiler_version == "11.2"]
  track_features:
    - cudatoolkit               # [cuda_compiler_version != "None"]
  string: cuda{{ cuda_compiler_version | replace('.', '') }}_py{{ CONDA_PY }}_h{{ PKG_HASH }}_{{ mpi_prefix }}_{{ PKG_BUILDNUM }}  # [cuda_compiler_version != "None"]
  string: cpu_py{{ CONDA_PY }}_h{{ PKG_HASH }}_{{ mpi_prefix }}_{{ PKG_BUILDNUM }} # [cuda_compiler_version == "None"]
  script_env:
{% if cuda_major == 11 %}
    - Kokkos_OPT_ARGS=-DKokkos_ARCH_KEPLER35=ON
{% endif %}
{% if cuda_major == 12 %}
    - Kokkos_OPT_ARGS=-DKokkos_ARCH_MAXWELL50=ON
{% endif %}


requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - {{ stdlib("c") }}
    - {{ compiler('cuda') }}    # [cuda_compiler_version != "None"]
    - rust
    - make
    - cmake=3.28
  host:
    - {{ mpi }}  # [mpi != 'nompi']
    - cuda-version {{ environ.get("cuda_compiler_version") }}.*  # [cuda_compiler_version != "None"]
    {% if cuda_major == 11 %}
    - cudatoolkit {{ environ.get("cuda_compiler_version") }}.*
    {% endif %}
    {% if cuda_major == 12 %}
    - nvidia/label/cuda-{{ environ.get("cuda_compiler_version") }}.*::cuda-toolkit # [cuda_compiler_version != "None"]
    - cuda-cudart-dev
    - cuda-driver-dev
    {% endif %}
    - libtorch  =*=cpu*  # [cuda_compiler_version == "None"]
    - libtorch  =*=cuda{{ cuda_compiler_version | replace('.', '') }}*  # [cuda_compiler_version != "None"]
    - fftw
    - fftw * {{ mpi_prefix }}_*
    - mkl # [target_platform == linux-64]
    - gsl
    - voro
    # https://github.com/lammps/lammps/blob/8389e2eb8074712b6850b3bf25fd3c3852e36f10/src/PLUMED/fix_plumed.cpp#L80-L82
    - plumed >=2.4,<2.10 # [target_platform == linux-64]
    - plumed * {{ mpi_prefix }}_*  # [target_platform == linux-64 and mpi != 'nompi' and build_platform == target_platform]
  run:
    - {{ mpi }}  # [mpi != 'nompi']
    - __cuda  # [cuda_compiler_version != "None"]
    - libtorch  =*=cpu*  # [cuda_compiler_version == "None"]
    - libtorch  =*=cuda{{ cuda_compiler_version | replace('.', '') }}*  # [cuda_compiler_version != "None"]

about:
  home: https://docs.metatensor.org/latest/index.html
  license: BSD-3-Clause
  license_family: BSD
  summary: 'Metatensor-enabled version of LAMMPS'
  description: |
    Metatensor-enabled version of LAMMPS
  doc_url: https://docs.metatensor.org/latest/atomistic/engines/lammps.html
  dev_url: https://github.com/metatensor/lammps

extra:
  recipe-maintainers:
    - abmazitov
```

The `build.sh` script is as follows:

```bash
#!/bin/bash

PLATFORM=$(uname)
args=""

if [[ "$PLATFORM" == 'Darwin' ]]; then
  BUILD_OMP=OFF
else
  BUILD_OMP=ON
  if [[ ${cuda_compiler_version} != "None" ]]; then
    CUDA_TOOLKIT_ROOT_DIR=$BUILD_PREFIX/targets/x86_64-linux
    args=$args" -DPKG_KOKKOS=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON ${Kokkos_OPT_ARGS} -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR "
  fi
  # PLUMED (for now only available on linux)
  args=$args" -DPKG_PLUMED=ON "
  if [[ ${mpi} != "nompi" ]]; then
    MPICC=mpicc
    MPICXX=mpicxx
    args=$args" -DPLUMED_CONFIG_CC=$MPICC -DPLUMED_CONFIG_CXX=$MPICXX "
  fi

fi

# Parallel and library
export LDFLAGS="-L$PREFIX/lib $LDFLAGS"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"

if [ "${mpi}" == "nompi" ]; then
  ENABLE_MPI=OFF
else
  ENABLE_MPI=TRUE
  export LDFLAGS="-lmpi ${LDFLAGS}"
fi


mkdir build && cd build

cmake -DPKG_ML-METATENSOR=ON \
      -DLAMMPS_INSTALL_RPATH=ON \
      -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
      -DPKG_REPLICA=ON \
      -DPKG_MC=ON \
      -DPKG_MOLECULE=ON \
      -DPKG_MISC=ON \
      -DPKG_KSPACE=ON \
      -DPKG_MANIFOLD=ON \
      -DPKG_QTB=ON \
      -DPKG_REACTION=ON \
      -DPKG_RIGID=ON \
      -DPKG_SHOCK=ON \
      -DPKG_SPIN=ON \
      -DPKG_VORONOI=ON \
      -DPKG_MPIIO=$ENABLE_MPI \
      -DPKG_EXTRA_PAIR=ON \
      -DBUILD_OMP=$BUILD_OMP \
      -DENABLE_MPI=$ENABLE_MPI \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH=$PREFIX/lib \
      $args \
      ../cmake

cmake --build . --parallel ${CPU_COUNT} -- VERBOSE=1
cmake --build . --target install

if [ "${mpi}" == "nompi" ]; then
  cp lmp ${PREFIX}/bin/lmp_serial
else
  cp lmp ${PREFIX}/bin/lmp_mpi
fi
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

In order to adapt the build recipe for your system, you need to check that the following
variables are set correctly:

- `Kokkos_OPT_ARGS=-DKokkos_ARCH_KEPLER35=ON` - check that here the correct version of the
    GPU architecture is set in `meta.yaml`. The available options can be found
    [here](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#nvidia-gpus).
- `mpi` - check that the correct MPI implementation is set in `conda_build_config.yaml`.
    The avaiable options are: `nompi`, `mpich`, `openmpi`.

The package can be built using the following command:

```bash
conda build lammps-kokkos
```

After this, the build process is started, and the package is built. This might take some time.
In the end, the built `.conda` bundle will be stored in the CONDA_PREFIX/conda-bld directory.
The exact location and the package name can be found in the output of the `conda build` 
This name will depend on the version of the LAMMPS, CUDA, Python, and MPI, so
mind that the exact name can be different in your case. 

To verify that your package was properly built and now available for the installation,
run:

```bash
conda search lammps-metatensor --use-local
```

This will use the local conda repository to search for the package. If the package
is found, you can proceed to the next step.

## Installing the package

To install the built package, run:

```bash
conda install lammps-metatensor --use-local
```

This will install the package into the current conda environment. You might need to
check that the package which is planned to install is the same, as the one you built.
Otherwise, copy the exact name of the package from the output of the `conda search`
command.

## Running KOKKOS-enabled LAMMPS-METATENSOR with PET-MAD

After the installation of the KOKKOS-enabled LAMMPS-METATENSOR, you can run the
simulation with PET-MAD. The procedure is the same as for the plain GPU-accelerated
LAMMPS, but involves a few changes in the input files to activate the KOKKOS-enabled
subroutines. 

The example of the input file **`lammps.in`** is as follows:

```bash
# LAMMPS input file with KOKKOS support
units metal
atom_style atomic/kk
read_data silicon.data
run_style verlet/kk
pair_style metatensor/kk pet-mad-latest.pt device cpu extensions extensions/
pair_coeff * * 14
neighbor 2.0 bin
timestep 0.001
dump myDump all xyz 10 trajectory.xyz
dump_modify myDump element Si
thermo_style multi
thermo 1
velocity all create 300 87287 mom yes rot yes
fix 1 all nvt temp 300 300 0.10
run 100
```

Next, create a **`silicon.data`** file (doesn't require any changes):

```bash
# LAMMPS data file for Silicon unit cell
8 atoms
1 atom types

0.0  5.43  xlo xhi
0.0  5.43  ylo yhi
0.0  5.43  zlo zhi

Masses

1  28.084999992775295 # Si

Atoms # atomic

1   1   0   0   0
2   1   1.3575   1.3575   1.3575
3   1   0   2.715   2.715
4   1   1.3575   4.0725   4.0725
5   1   2.715   0   2.715
6   1   4.0725   1.3575   4.0725
7   1   2.715   2.715   0
8   1   4.0725   4.0725   1.3575
```

Finally, run the simulation:

```bash
lmp_serial -k on g 1 -pk kokkos newton on -in lammps.in  # Serial version   
mpirun -np 1 lmp_mpi -k on g 1 -pk kokkos newton on -in lammps.in  # MPI version
```

Here, the `-k on g 1 -pk kokkos newton on` flags are used to activate the KOKKOS
subroutines. Specifically `g 1` is used to specify, how many GPUs are the simulation is
parallelized over, so if running the large systems on two or more GPUs, this number
should be adjusted accordingly.

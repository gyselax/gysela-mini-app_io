# Gysela Mini Apps

A minimal application demonstrating GYSELA I/O operations and testing the CPU performance scaling for 5D particle distribution functions.

This repository contains mini apps that allow easy coupling with GyselaX++.

## Installing

```bash
git clone git@github.com:gyselax/gysela-mini-app_io.git
cd gysela-mini-app_io
git submodule update --init --recursive
```

## Quick install

After cloning and initializing submodules:

```bash
sh ./installer.sh <MACHINE>
```

Example on Persee (CPU): `./installer.sh persee/xeon`


`<MACHINE>` is a folder under `src/external/gyselalibxx/toolchains/`. 
Available values:

- `a100.leonardo.spack` — Leonardo (A100)
- `a100.raven.spack` — Raven (A100)
- `docker.gyselalibxx_env` — Docker / CI
- `genoa.gcc.adastra.spack` — Adastra (Genoa CPU)
- `h100.jean-zay.spack` — Jean Zay (H100)
- `mi250.hipcc.adastra.spack` — Adastra (MI250)
- `persee/v100` — Persee (V100)
- `persee/xeon` — Persee (CPU); default on Persee hosts if `<MACHINE>` is omitted
- `v100.ruche` — Ruche (V100)

## Manual installation

```bash
source src/external/gyselalibxx/toolchains/<MACHINE>/environment.sh
python -m venv .gys_env    # skip if .gys_env already exists
source .gys_env/bin/activate
pip install -e ".[dev]"
```

For more details see [Gyselalib++ environment toolchains](https://gyselax.github.io/gyselalibxx/toolchains/index.html#environment-setup).

## Building

By default, both apps are built:

- IO app
- Compression app

You can disable either app at configuration time using CMake options:

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=src/external/gyselalibxx/toolchains/<MACHINE>/toolchain.cmake 
cmake --build build -j 4
```

For the docker toolchain, you should use the following in the docker container:
```bash
cmake -S . -B build
cmake --build build -j 4
```

## Python venv

In order to use the python tools, you'll need to execute the following commands from the repo's root:
```bash
source .gys_env/bin/activate
```

If you want to use Python insitu-diagnostics, the following commands are available from the command line:
```bash
read-timing-stats
verify-fluid-moments
```

## Running

Each app has its own usage instructions. See the README file in the corresponding app folder for details.

The compression mini-app (`apps/compression/`) takes a GYSELA YAML config and `pdi_out.yaml` on the command line. Case-specific templates are `params_landau_damping.yaml` and `params_two_stream.yaml`; the compression benchmark launcher uses `params_landau_damping.yaml` by default.


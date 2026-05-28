# Gysela Mini Apps

A minimal application demonstrating GYSELA I/O operations and testing the CPU performance scaling for 5D particle distribution functions.

This repository contains mini apps that allow easy coupling with GyselaX++.

## Installing

```bash
git clone git@github.com:gyselax/gysela-mini-app_io.git
cd gysela-mini-app_io
git submodule update --init --recursive
```

## Sourcing the environment

```bash
source external/gyselalibxx/toolchains/<MACHINE>/environment.sh
```

For more details see [Gyselalib++ environment toolchains](https://gyselax.github.io/gyselalibxx/toolchains/index.html#environment-setup).

## Building

By default, both apps are built:

- IO app
- Compression app

You can disable either app at configuration time using CMake options:

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=external/gyselalibxx/toolchains/<MACHINE>/toolchain.cmake \
  -DBUILD_IO_APP=OFF \
  -DBUILD_COMPRESSION_APP=ON
cmake --build build -j 4
```
If you want to use Python insitu-diagnostics set additionally the `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/your/repo/gysela-mini-app_io/python:$PYTHONPATH
```

## Running

Each app has its own usage instructions. See the README file in the corresponding app folder for details.


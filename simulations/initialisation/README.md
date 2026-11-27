# fdistribu initialisation write benchmark

Finds the fastest way to write the initial Gysela-X distribution function
to disk. Each MPI rank computes the same Maxwellian as the Gysela-X initialisation 
on its own block of `fdistribu(species, tor1, tor2[, tor3], vpar, mu)` and streams
it to disk piece by piece.

Pure Python (mpi4py + numexpr + h5py / netCDF4 / MPI-IO).

## What is compared

| `method`                   | library        | file layout       | how the bytes reach the file                         |
|----------------------------|----------------|-------------------|------------------------------------------------------|
| `hdf5-shared-collective`   | parallel HDF5  | one shared file   | `H5Dwrite`, MPI-IO collective (two-phase aggregation) |
| `hdf5-shared-independent`  | parallel HDF5  | one shared file   | `H5Dwrite`, MPI-IO independent                        |
| `hdf5-fpp`                 | HDF5           | one file per rank | serial `H5Dwrite` (sec2 driver)                       |
| `netcdf-shared-collective` | netCDF-4       | one shared file   | `nc_put_vara`, collective                             |
| `netcdf-shared-independent`| netCDF-4       | one shared file   | `nc_put_vara`, independent                            |
| `netcdf-fpp`               | netCDF-4       | one file per rank | serial `nc_put_vara`                                  |
| `raw-shared-collective`    | custom (MPI-IO)| one shared file   | `MPI_File_write_all` (aggregation)                    |
| `raw-shared-independent`   | custom (MPI-IO)| one shared file   | `MPI_File_write` through a subarray file view         |
| `raw-shared-posix`         | custom (POSIX) | one shared file   | `pwrite` of each contiguous run                       |
| `raw-fpp`                  | custom (POSIX) | one file per rank | `pwrite`                                              |

The raw files are plain C-ordered `float64`. A `fdistribu.json` file holds the
shape, dims, coordinates and each rank's block. A file can then
be opened with `np.memmap(path, dtype='<f8', shape=global_shape)`.

Shared files use the same layout and coordinates as the Gysela-X `fdistribu`
dataset, stored contiguously. File-per-process files hold the local block, plus
`global_shape`/`local_start` attributes for reassembly.

### Write path, and what makes it fast

- **Streaming, bounded memory**: a rank never holds more than `--piece-mb`
  (default 512 MiB) of fdistribu. It computes a piece with numexpr straight into
  a reused buffer, then writes it.
- **Contiguous blocks**: ranks are decomposed over the outermost axes first
  (species, then tor1, then tor2). Each rank's block is then one contiguous
  region of the shared file.
- **No double write**: HDF5 datasets use `H5D_FILL_TIME_NEVER` and early
  allocation, and netCDF uses `set_fill_off()`. Without these settings the
  library first writes the whole file as fill values.
- **Lustre**: before each test the output directory is striped
  (`lfs setstripe -c <stripe_count> -S <stripe_size_mb>M`). HDF5 objects are
  aligned on the stripe size, and HDF5 metadata ops are collective.

## Metrics

All times are measured on the slowest rank, between barriers.

| metric              | meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `bw_synced_GBps`    | **headline**: data / (compute + write + close + `fsync`). Data is really on the OSTs. |
| `bw_write_GBps`     | data / (open + write calls + close). Can be flattered by the Lustre client cache. |
| `time_total_synced` | wall time of the whole initialisation, `fsync` included                 |
| `time_write`        | open + write calls + close, slowest rank                                |
| `time_compute`      | Maxwellian evaluation only, slowest rank                                |
| `time_sync`         | `fsync` of the written files                                            |
| `mem_peak_rank_MB`  | peak resident memory (RSS high-water mark) of the largest rank          |
| `mem_peak_node_GB`  | sum of the per-rank peaks on the most loaded node                       |
| `mem_peak_total_GB` | sum of the per-rank peaks over the whole job                            |
| `mem_baseline_rank_MB` | per-rank RSS before the piece buffer exists (Python + MPI + I/O libraries) |
| `piece_MB`          | size of the largest piece buffer actually used                          |
| `disk_GB`           | bytes on disk (`st_size`), summed over the fdistribu files                |
| `disk_alloc_GB`     | bytes actually allocated (`st_blocks`); less than the payload means holes |
| `size_errors`       | files smaller than the payload they hold (must be 0)                      |
| `verify_points`     | values read back with the same library, over all ranks                    |
| `verify_errors`     | read-back values that differ from the Maxwellian (must be 0)              |

### Checks that the data reached the disk

After the `fsync`, every run checks the files it wrote. These checks are not timed.

1. **Size, always on.** Each per-rank file must be at least as large as its
   local block, and the shared file at least as large as all of fdistribu. HDF5
   and netCDF only add metadata on top of that.
2. **Read-back** (`--verify N`, default 8). Each rank reopens its file with the
   same library and checks the first and last value of every piece it wrote,
   plus N random points, against the Maxwellian.

If a file is too small or a value is wrong, the run prints `### FAILED`, still
prints its results line, and exits with a non-zero status. srun then fails, so
IOPS marks the test as failed.

Memory is measured with `getrusage` after the write, before
verification. It includes everything the libraries hold (MPI-IO collective
buffers, HDF5/netCDF caches). The node and job values add up per-rank peaks, so
they are an upper bound on what was resident at one instant. In practice the
ranks peak together, while each holds its piece buffer.

On the medium grid, 2 nodes have far more RAM than one file holds. Without the
`fsync`, file-per-process writes mostly measure page-cache speed.

## Running on the cluster

The repository must be on Lustre (the per-test data is written under `output/`).
From the repository root run:

```bash
source toolchains/adastra/genoa/environment.sh && source .gys_env/bin/activate

# 2. check that the Python stack is the parallel one (both must print True)
python -c "import h5py, netCDF4, mpi4py, numexpr; print(h5py.get_config().mpi, bool(netCDF4.__has_parallel4_support__))"

# 3. run, from the repository root
iops check simulations/initialisation/iops_config.yaml
iops run   simulations/initialisation/iops_config.yaml --dry-run
iops run   simulations/initialisation/iops_config.yaml (--use-cache)
```

IOPS submits a single `sbatch` (2 GENOA nodes) and runs every test inside it
with `srun`. If the allocation runs out of time, rerun with `--use-cache`: tests
that already succeeded are skipped. Each test deletes its data when it ends.

Results go to `output/run_XXX/results_init_bench.csv` and
`init_bench_report.html`. Each test's log is also recorded.

### Default sweep

`backend × layout × io_mode` (10 methods) × `stripe_count ∈ {16, -1}` (shared files only;
file-per-process uses 1 OST per file) × `nodes ∈ {1, 2}` × 3 repetitions, on the
medium grid (68 GB). That is 34 tests and 102 runs.

Knobs in `vars:`:

| var             | default    | notes                                                              |
|-----------------|------------|--------------------------------------------------------------------|
| `grid`          | `medium`   | `small` (0.5 GB, debugging), `large` (275 GB). Large takes about 4× longer: narrow the methods first. |
| `stripe_count`  | `16, -1`   | shared files only                                                  |
| `piece_mb`      | `512`      | memory per rank for one compute + write piece                      |
| `nc_format`     | `NETCDF4`  | `NETCDF3_64BIT_DATA`goes through PnetCDF, which needs netcdf-c `+parallel-netcdf` |
| `mpi_hints`     | `""`       | for example `romio_cb_write=enable,cb_nodes=4`. Cray MPICH also reads `MPICH_MPIIO_HINTS`. |

## Summary plots

```bash
python processing/initialisation/plot_summary.py output/run_001/results_init_bench.csv -o figures/
```

This writes three figures and a CSV, using the same layout as the HTML summary:

- `bandwidth_vs_nodes.png`: bandwidth against node count for the best configuration of each family
- `time_split.png`: compute, write and fsync time at the largest node count
- `configurations_table.png` and `configurations_table.csv`: every configuration, with time and bandwidth per node count and the speed-up

Repetitions are averaged, and the bandwidth plot shows their min–max range.
Use `--grid` when the CSV holds several grids, and `--format pdf` for vector output.

## Files

- `processing/initialisation/init_bench.py`: decomposition, Maxwellian, timing loop, results line
- `processing/initialisation/writers.py`: one writer class per backend, each handling both layouts
- `processing/initialisation/plot_summary.py`: figures and table from the IOPS results CSV
- `simulations/initialisation/iops_config.yaml`: IOPS sweep, single-allocation Slurm setup, parser, report

// SPDX-License-Identifier: MIT
#include <mpi.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <ddc/ddc.hpp>

#include <paraconf.h>
#include <pdi.h>

#include "ddc_alias_inline_functions.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "mpilayout.hpp"
#include "paraconfpp.hpp"
#include "pdi_helper.hpp"

using std::cout;
using std::endl;
using std::string;
using std::chrono::steady_clock;

namespace fs = std::filesystem;

namespace {

// When true, everything the OS still holds in its page cache is pushed to the
// file system inside the timed region. Without it, a plain POSIX close() (as
// done for one-file-per-rank output) returns before Lustre has the data, which
// would make that method look faster than it really is. MPI_File_close, used by
// the shared-file and subfiling paths, already synchronises.
constexpr bool flush_to_file_system = true;

struct ConfigHandles {
  PC_tree_t conf_gyselax;
  PC_tree_t conf_pdi;
};

void display_help(std::string const &exe) {
  std::cerr << "usage: " << exe << " <config_file.yaml> <pdi_config.yaml>"
            << endl;
  std::exit(EXIT_FAILURE);
}

ConfigHandles parse_config_files(int argc, char **argv) {
  ConfigHandles configs{};
  std::string exe = argv[0];
  if (argc > 2) {
    fs::path gysela_config_yml = argv[1];
    if (gysela_config_yml.extension() != ".yaml" &&
        gysela_config_yml.extension() != ".yml") {
      std::cerr << "Expected a .yaml file for the config_file.yaml. Received : "
                << gysela_config_yml << endl;
      display_help(exe);
    }
    configs.conf_gyselax = PC_parse_path(gysela_config_yml.c_str());
    fs::path pdi_config_yml = argv[2];
    if (pdi_config_yml.extension() != ".yaml" &&
        pdi_config_yml.extension() != ".yml") {
      std::cerr << "Expected a .yaml file for the pdi_config.yaml. Received : "
                << pdi_config_yml << endl;
      display_help(exe);
    }
    configs.conf_pdi = PC_parse_path(pdi_config_yml.c_str());
  } else {
    display_help(exe);
  }
  PC_errhandler(PC_NULL_HANDLER);
  return configs;
}

void print_banner(int rank) {
  if (rank != 0) {
    return;
  }
  cout << "==========================================" << endl;
  cout << "        GYSELA MINI APP - NETCDF          " << endl;
  cout << "==========================================" << endl;
}

IdxRangeSp init_species_from_yaml(PC_tree_t conf_gyselax) {
  int const nb_species = PCpp_len(conf_gyselax, ".SpeciesInfo");
  IdxRangeSp idx_range_sp(IdxSp(0), IdxStepSp(nb_species));

  host_t<DFieldMemSp> kinetic_charges(idx_range_sp);
  host_t<DFieldMemSp> kinetic_masses(idx_range_sp);
  for (int i = 0; i < nb_species; ++i) {
    std::string const base = ".SpeciesInfo[" + std::to_string(i) + "]";
    kinetic_charges(IdxStepSp(i)) =
        PCpp_double(conf_gyselax, (base + ".charge").c_str());
    kinetic_masses(IdxStepSp(i)) =
        PCpp_double(conf_gyselax, (base + ".mass").c_str());
  }

  ddc::init_discrete_space<Species>(std::move(kinetic_charges),
                                    std::move(kinetic_masses));
  return idx_range_sp;
}

IdxRangeSpTor3DV2D initialise_mesh(int rank, PC_tree_t conf_gyselax) {
  IdxRangeSp const idx_range_sp = init_species_from_yaml(conf_gyselax);
  if (rank == 0) {
    cout << "Number of kinetic species: " << idx_range_sp.size() << endl;
  }

  IdxRange<GridTor1> const idx_range_tor1 =
      init_spline_dependent_idx_range<GridTor1, BSplinesTor1,
                                      SplineInterpPointsTor1>(conf_gyselax,
                                                              "Tor1");
  IdxRange<GridTor2> const idx_range_tor2 =
      init_spline_dependent_idx_range<GridTor2, BSplinesTor2,
                                      SplineInterpPointsTor2>(conf_gyselax,
                                                              "Tor2");
  IdxRange<GridTor3> const idx_range_tor3 =
      init_spline_dependent_idx_range<GridTor3, BSplinesTor3,
                                      SplineInterpPointsTor3>(conf_gyselax,
                                                              "Tor3");
  IdxRange<GridVpar> const idx_range_vpar =
      init_spline_dependent_idx_range<GridVpar, BSplinesVpar,
                                      SplineInterpPointsVpar>(conf_gyselax,
                                                              "Vpar");
  IdxRange<GridMu> const idx_range_mu =
      init_spline_dependent_idx_range<GridMu, BSplinesMu, SplineInterpPointsMu>(
          conf_gyselax, "Mu");

  if (rank == 0) {
    cout << "Grid sizes:" << endl;
    cout << "  tor1: " << idx_range_tor1.size() << endl;
    cout << "  tor2: " << idx_range_tor2.size() << endl;
    cout << "  tor3: " << idx_range_tor3.size() << endl;
    cout << "  vpar: " << idx_range_vpar.size() << endl;
    cout << "  mu: " << idx_range_mu.size() << endl;
  }

  return IdxRangeSpTor3DV2D(idx_range_sp, idx_range_tor1, idx_range_tor2,
                            idx_range_tor3, idx_range_vpar, idx_range_mu);
}

void init_distribution_fun(DFieldSpGrid allfdistribu,
                           IdxRangeSpTor3DV2D const &mesh,
                           PC_tree_t conf_gyselax) {
  // Read species-specific parameters from YAML
  IdxRangeSp const idx_range_sp(mesh);

  int const nb_species = idx_range_sp.size();
  host_t<DFieldMem<IdxRangeSp>> density_host(idx_range_sp);
  host_t<DFieldMem<IdxRangeSp>> temperature_host(idx_range_sp);
  host_t<DFieldMem<IdxRangeSp>> mean_velocity_host(idx_range_sp);

  for (int i = 0; i < nb_species; ++i) {
    std::string const base = ".SpeciesInfo[" + std::to_string(i) + "]";
    density_host(IdxSp(i)) =
        PCpp_double(conf_gyselax, (base + ".N_maxw").c_str());
    temperature_host(IdxSp(i)) =
        PCpp_double(conf_gyselax, (base + ".T_maxw").c_str());
    mean_velocity_host(IdxSp(i)) =
        PCpp_double(conf_gyselax, (base + ".Upar_maxw").c_str());
  }

  // Copy to device
  DFieldMem<IdxRangeSp> density_alloc(idx_range_sp);
  DFieldMem<IdxRangeSp> temperature_alloc(idx_range_sp);
  DFieldMem<IdxRangeSp> mean_velocity_alloc(idx_range_sp);
  DField<IdxRangeSp> density = get_field(density_alloc);
  DField<IdxRangeSp> temperature = get_field(temperature_alloc);
  DField<IdxRangeSp> mean_velocity = get_field(mean_velocity_alloc);
  ddc::parallel_deepcopy(density, density_host);
  ddc::parallel_deepcopy(temperature, temperature_host);
  ddc::parallel_deepcopy(mean_velocity, mean_velocity_host);

  // Compute Maxwellian distribution: fM(vpar,mu) =
  // (2*PI*T)**(-1.5)*n*exp(-energy) with energy = 0.5*((vpar-Upar)**2-mu) /T
  ddc::parallel_for_each(
      Kokkos::DefaultExecutionSpace(), mesh,
      KOKKOS_LAMBDA(IdxSpTor3DV2D const ispgrid) {
        IdxSp const isp(ispgrid);
        double const density_loc = density(isp);
        double const temperature_loc = temperature(isp);
        double const mean_velocity_loc = mean_velocity(isp);
        double const vpar = ddc::coordinate(ddc::select<GridVpar>(ispgrid));
        double const mu = ddc::coordinate(ddc::select<GridMu>(ispgrid));
        double const inv_2piT = 1. / (2. * M_PI * temperature_loc);
        double const coeff_maxw = Kokkos::pow(inv_2piT, 1.5);
        double const energy =
            0.5 *
            ((vpar - mean_velocity_loc) * (vpar - mean_velocity_loc) + mu) /
            temperature_loc;
        allfdistribu(ispgrid) = coeff_maxw * density_loc * Kokkos::exp(-energy);
      });
}

void write_fdistribu(int rank, IdxRangeSpTor3DV2D const &local_mesh,
                     IdxRangeSpTor3DV2D const &global_mesh,
                     host_t<DFieldMemSpGrid> const &allfdistribu_host) {
  if (rank == 0) {
    cout << "Writing 5D distribution function to file(s)." << endl;
  }

  // Expose the local index range (starts + extents) of this rank's chunk, used
  // by the PDI configuration for the hyperslab selection and per-rank file names.
  PDI_expose_idx_range(local_mesh, "local_fdistribu");

  // Global extents of the distribution function.
  std::array<std::size_t, 1> species_extents_arr = {
      static_cast<std::size_t>(global_mesh.template extent<Species>())};
  std::array<std::size_t, 1> tor1_extents_arr = {static_cast<std::size_t>(
      global_mesh.template extent<GridTor1>().value())};
  std::array<std::size_t, 1> tor2_extents_arr = {static_cast<std::size_t>(
      global_mesh.template extent<GridTor2>().value())};
  std::array<std::size_t, 1> tor3_extents_arr = {static_cast<std::size_t>(
      global_mesh.template extent<GridTor3>().value())};
  std::array<std::size_t, 1> vpar_extents_arr = {static_cast<std::size_t>(
      global_mesh.template extent<GridVpar>().value())};
  std::array<std::size_t, 1> mu_extents_arr = {static_cast<std::size_t>(
      global_mesh.template extent<GridMu>().value())};
  PDI_expose("species_extents", species_extents_arr.data(), PDI_OUT);
  PDI_expose("tor1_extents", tor1_extents_arr.data(), PDI_OUT);
  PDI_expose("tor2_extents", tor2_extents_arr.data(), PDI_OUT);
  PDI_expose("tor3_extents", tor3_extents_arr.data(), PDI_OUT);
  PDI_expose("vpar_extents", vpar_extents_arr.data(), PDI_OUT);
  PDI_expose("mu_extents", mu_extents_arr.data(), PDI_OUT);

  // Expose the local 6D chunk and trigger the write event. The PDI
  // configuration decides how it is written (plugin, shared file, one file per
  // rank, subfiling).
  {
    ddc::PdiEvent event("write_fdistribu");
    event.with("local_fdistribu_sptor3Dv2D", allfdistribu_host);
  }

  if (rank == 0) {
    cout << "5D distribution function written successfully." << endl;
  }
}

} // namespace

int main(int argc, char **argv) {
  //---------------------------------------------------------
  // Read the configuration
  //---------------------------------------------------------
  ConfigHandles configs = parse_config_files(argc, argv);
  bool const subfiling =
      PCpp_bool(configs.conf_gyselax, ".Application.subfiling");

  //---------------------------------------------------------
  // MPI initialisation. HDF5 subfiling needs MPI_THREAD_MULTIPLE (on Cray MPICH
  // this also requires MPICH_MAX_THREAD_SAFETY=multiple in the environment).
  // Every other strategy keeps the cheaper single-threaded level.
  //---------------------------------------------------------
  int const required = subfiling ? MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
    std::fprintf(stderr, "provided level = %d, required level = %d\n",
                 provided, required);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  Kokkos::ScopeGuard scope(argc, argv);
  ddc::ScopeGuard ddc_scope(argc, argv);

  steady_clock::time_point const time_start = steady_clock::now();

  print_banner(rank);
  if (rank == 0) {
    cout << "Subfiling: " << (subfiling ? "enabled" : "disabled")
         << " (MPI thread level provided: " << provided << ")" << endl;
  }
  PDI_init(configs.conf_pdi);

  //---------------------------------------------------------
  // Initialisation of the global and local (MPI distributed) mesh
  //---------------------------------------------------------
  IdxRangeSpTor3DV2D const global_mesh =
      initialise_mesh(rank, configs.conf_gyselax);
  IdxRangeSpTor3DV2D const local_mesh =
      Tor3DSplit::distribute_idx_range(global_mesh, comm_size, rank);

  //---------------------------------------------------------
  // Initialisation of the distribution function
  //---------------------------------------------------------
  DFieldMemSpGrid allfdistribu(local_mesh);
  init_distribution_fun(get_field(allfdistribu), local_mesh,
                        configs.conf_gyselax);

  // Host copy of the local 6D distribution function.
  host_t<DFieldMemSpGrid> allfdistribu_host(local_mesh);
  ddc::parallel_deepcopy(allfdistribu_host, allfdistribu);

  //---------------------------------------------------------
  // Write the distribution function using PDI (decl_netcdf / decl_hdf5)
  //
  // All ranks start the clock together (barrier), every rank times its own
  // write, and the reported value is the SLOWEST rank.
  //
  // ::sync() blocks until this node's cached data is written out (flushes data to the filesystem).
  //---------------------------------------------------------
  MPI_Barrier(MPI_COMM_WORLD);
  steady_clock::time_point const time_write_start = steady_clock::now();
  write_fdistribu(rank, local_mesh, global_mesh, allfdistribu_host);
  if (flush_to_file_system) {
    ::sync();
  }
  steady_clock::time_point const time_end = steady_clock::now();

  double const time_write_local =
      std::chrono::duration<double>(time_end - time_write_start).count();
  double const time_total_local =
      std::chrono::duration<double>(time_end - time_start).count();
  double time_write = 0.;
  double time_total = 0.;
  MPI_Reduce(&time_write_local, &time_write, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&time_total_local, &time_total, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    cout << "Time write: " << time_write << "s" << endl;
    cout << "Time total: " << time_total << "s" << endl;
  }

  PC_tree_destroy(&configs.conf_pdi);
  PC_tree_destroy(&configs.conf_gyselax);
  PDI_finalize();
  MPI_Finalize();

  return EXIT_SUCCESS;
}

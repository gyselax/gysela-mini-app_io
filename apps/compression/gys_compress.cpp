// SPDX-License-Identifier: MIT
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include <paraconf.h>
#include <pdi.h>

#include "spline_definitions_xyvxvy.hpp"

#include "bsl_advection_vx.hpp"
#include "bsl_advection_x.hpp"
#include "chargedensitycalculator.hpp"
#include "ddc_alias_inline_functions.hpp"
#include "ddc_helper.hpp"
#include "fft_poisson_solver.hpp"
#include "geometry_xyvxvy.hpp"
#include "input.hpp"
#include "maxwellianequilibrium.hpp"
#include "mpichargedensitycalculator.hpp"
#include "mpisplitvlasovsolver.hpp"
#include "mpitransposealltoall.hpp"
#include "neumann_spline_quadrature.hpp"
#include "output.hpp"
#include "paraconfpp.hpp"
#include "params.yaml.hpp"
#include "pdi_out.yml.hpp"
#include "predcorr_compress.hpp"
#include "qnsolver.hpp"
#include "singlemodeperturbinitialisation.hpp"
#include "species_info.hpp"
#include "species_init.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace {

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

void init_landau_damping(
    IdxRangeSp const idx_range_kinsp,
    PC_tree_t const& conf_gyselax,
    DFieldMemSpXYVxVy& allfdistribu_x2D_split,
    DFieldMemSpVxVy& allfequilibrium) {
  MaxwellianEquilibrium const init_fequilibrium =
      MaxwellianEquilibrium::init_from_input(idx_range_kinsp, conf_gyselax);
  init_fequilibrium(get_field(allfequilibrium));

  SingleModePerturbInitialisation const init =
      SingleModePerturbInitialisation::init_from_input(
          get_const_field(allfequilibrium), idx_range_kinsp, conf_gyselax);
  init(get_field(allfdistribu_x2D_split));
}

void init_two_stream(
    IdxRangeSp const idx_range_kinsp,
    PC_tree_t const& conf_gyselax,
    DFieldMemSpXYVxVy& allfdistribu_x2D_split,
    DFieldMemSpVxVy& allfequilibrium) {
  IdxRangeXYVxVy const gridxyvxvy =
      get_idx_range<GridX, GridY, GridVx, GridVy>(allfdistribu_x2D_split);
  IdxRangeXY const gridxy = get_idx_range<GridX, GridY>(allfdistribu_x2D_split);
  IdxRangeVxVy const gridvxvy = get_idx_range<GridVx, GridVy>(allfdistribu_x2D_split);

  DFieldSpXYVxVy allfdistribu = get_field(allfdistribu_x2D_split);
  DFieldSpVxVy allfequilibrium_field = get_field(allfequilibrium);

  DFieldMemXY perturbation_alloc(gridxy);
  DFieldXY perturbation = get_field(perturbation_alloc);

  double const inv_2pi = 1. / (2. * M_PI);
  double const length_x = PCpp_double(conf_gyselax, ".SplineMesh.x_max") -
                           PCpp_double(conf_gyselax, ".SplineMesh.x_min");
  double const length_y = PCpp_double(conf_gyselax, ".SplineMesh.y_max") -
                           PCpp_double(conf_gyselax, ".SplineMesh.y_min");

  ddc::host_for_each(idx_range_kinsp, [&](IdxSp const isp) {
    PC_tree_t const conf_isp = PCpp_get(conf_gyselax, ".SpeciesInfo[%d]", isp.uid());

    double const v0 = PCpp_double(conf_isp, ".mean_velocity_eq");
    double const eps = PCpp_double(conf_isp, ".perturb_amplitude");
    int const perturb_mode =
        static_cast<int>(PCpp_int(conf_isp, ".perturb_mode"));
    double const kx = perturb_mode * 2. * M_PI / length_x;
    double const ky = 0;
    // perturb_mode * 2. * M_PI / length_y;

    if (isp.uid() == 0) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) {
        cout << "two_stream kx = " << kx << endl;
      }
    }
    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        gridvxvy,
        KOKKOS_LAMBDA(IdxVxVy const ivxvy) {
          double const vx = ddc::coordinate(ddc::select<GridVx>(ivxvy));
          double const vy = ddc::coordinate(ddc::select<GridVy>(ivxvy));
          double const m1 = Kokkos::exp(
              -((vx - v0) * (vx - v0) + (vy ) * (vy )) / 2.);
          double const m2 = Kokkos::exp(
              -((vx + v0) * (vx + v0) + (vy ) * (vy )) / 2.);
          allfequilibrium_field(isp, ivxvy) = 0.5 * inv_2pi * (m1 + m2);
        });

    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        gridxy,
        KOKKOS_LAMBDA(IdxXY const ixy) {
          IdxX const ix = ddc::select<GridX>(ixy);
          IdxY const iy = ddc::select<GridY>(ixy);
          double const x = ddc::coordinate(ix);
          double const y = ddc::coordinate(iy);
          perturbation(ix, iy) =
              1. + eps * Kokkos::cos(kx * x) * Kokkos::cos(ky * y);
        });

    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        gridxyvxvy,
        KOKKOS_LAMBDA(IdxXYVxVy const ixyvxvy) {
          IdxX const ix = ddc::select<GridX>(ixyvxvy);
          IdxY const iy = ddc::select<GridY>(ixyvxvy);
          IdxVx const ivx = ddc::select<GridVx>(ixyvxvy);
          IdxVy const ivy = ddc::select<GridVy>(ixyvxvy);

          double fdistribu_val =
              perturbation(ix, iy) * allfequilibrium_field(isp, ivx, ivy);
          if (fdistribu_val < 1.e-60) {
            fdistribu_val = 1.e-60;
          }
          allfdistribu(isp, ix, iy, ivx, ivy) = fdistribu_val;
        });
  });
}

void init_case(
    IdxRangeSp const idx_range_kinsp,
    ConfigHandles const& configs,
    MPITransposeAllToAll<X2DSplit, V2DSplit>& transpose,
    DFieldMemSpVxVy& allfequilibrium,
    DFieldMemSpXYVxVy& allfdistribu_x2D_split,
    DFieldMemSpVxVyXY& allfdistribu_v2D_split) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string const case_name =
    PC_status(PC_get(configs.conf_gyselax, ".Input.case"))? "landau_damping": PCpp_string(configs.conf_gyselax, ".Input.case");
    if (rank == 0) {  
        cout << "case: " << case_name << endl;
      }
  if (case_name == "landau_damping") {
    init_landau_damping(idx_range_kinsp, configs.conf_gyselax,
                        allfdistribu_x2D_split, allfequilibrium);
  } else if (case_name == "two_stream") {
    init_two_stream(idx_range_kinsp, configs.conf_gyselax, allfdistribu_x2D_split,
                    allfequilibrium);
  } else {
    assert(false && "Unknown case");
  }

  transpose(Kokkos::DefaultExecutionSpace(), get_field(allfdistribu_v2D_split),
            get_const_field(allfdistribu_x2D_split));
}

} // namespace

void print_banner(int rank) {
  if (rank != 0) {
    return;
  }
  cout << "==========================================" << endl;
  cout << "      GYSELA COMPRESSION MINI APP         " << endl;
  cout << "==========================================" << endl;
}

int main(int argc, char **argv) {
  ConfigHandles configs = parse_config_files(argc, argv);

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  print_banner(rank);

  steady_clock::time_point time_points[6];
  std::vector<std::string> timing_names(6);

  time_points[0] = steady_clock::now();

  Kokkos::ScopeGuard scope(argc, argv);
  ddc::ScopeGuard ddc_scope(argc, argv);

  PDI_init(configs.conf_pdi);

  // Reading config
  // --> Mesh info
  IdxRangeX const idxrange_x =
      init_spline_dependent_idx_range<GridX, BSplinesX, SplineInterpPointsX>(
          configs.conf_gyselax, "x");
  IdxRangeY const idxrange_y =
      init_spline_dependent_idx_range<GridY, BSplinesY, SplineInterpPointsY>(
          configs.conf_gyselax, "y");
  IdxRangeVx const idxrange_vx =
      init_spline_dependent_idx_range<GridVx, BSplinesVx, SplineInterpPointsVx>(
          configs.conf_gyselax, "vx");
  IdxRangeVy const idxrange_vy =
      init_spline_dependent_idx_range<GridVy, BSplinesVy, SplineInterpPointsVy>(
          configs.conf_gyselax, "vy");
  IdxRangeXY const idxrange_xy(idxrange_x, idxrange_y);
  IdxRangeVxVy idxrange_vxvy(idxrange_vx, idxrange_vy);
  IdxRangeXYVxVy const idxrange_xyvxvy(idxrange_x, idxrange_y, idxrange_vx,
                                       idxrange_vy);

  IdxRangeSp const idx_range_kinsp = init_species(configs.conf_gyselax);
  IdxRangeSpXYVxVy const idxrange_glob_spxyvxvy(idx_range_kinsp,
                                                idxrange_xyvxvy);

  MPITransposeAllToAll<X2DSplit, V2DSplit> transpose(idxrange_glob_spxyvxvy,
                                                     MPI_COMM_WORLD);

  IdxRangeSpXYVxVy idxrange_spxyvxvy_x2Dsplit(
      transpose.get_local_idx_range<X2DSplit>());
  IdxRangeSpVxVyXY idxrange_spvxvyxy_v2Dsplit(
      transpose.get_local_idx_range<V2DSplit>());

  IdxRangeVxVy idxrange_vxvy_v2Dsplit(idxrange_spvxvyxy_v2Dsplit);
  IdxRangeVxVyXY idxrange_vxvyxy_v2Dsplit(idxrange_spvxvyxy_v2Dsplit);
  IdxRangeXYVxVy idxrange_xyvxvy_x2Dsplit(idxrange_spxyvxvy_x2Dsplit);

  SplineInterpolatorX const interpolator_x(idxrange_x);
  SplineInterpolatorY const interpolator_y(idxrange_y);
  SplineInterpolatorVx const interpolator_vx(idxrange_vx);
  SplineInterpolatorVy const interpolator_vy(idxrange_vy);

  IdxRangeSpVxVy idxrange_spvxvy_local(idxrange_spxyvxvy_x2Dsplit);

  // -------------------------------------------------------------------------
  // --> RESTART LOGIC & CONFIGURATION READ
  // -------------------------------------------------------------------------
  std::string fdistribu_filename = "none";
  if (!PC_status(PC_get(configs.conf_gyselax, ".Input.fdistribu_filename"))) {
    fdistribu_filename =
        PCpp_string(configs.conf_gyselax, ".Input.fdistribu_filename");
  }
  int nb_restart = 0;
  if (!PC_status(PC_get(configs.conf_gyselax, ".Input.nb_restart"))) {
    nb_restart =
        static_cast<int>(PCpp_int(configs.conf_gyselax, ".Input.nb_restart"));
  }
  int iter_offset = 0;
  if (!PC_status(PC_get(configs.conf_gyselax, ".Input.iter_offset"))) {
    iter_offset =
        static_cast<int>(PCpp_int(configs.conf_gyselax, ".Input.iter_offset"));
  }
  int compression_period = 0;
  if (!PC_status(PC_get(configs.conf_gyselax, ".CompressionBenchmark.compression_period"))) {
    compression_period = static_cast<int>(
        PCpp_int(configs.conf_gyselax, ".CompressionBenchmark.compression_period"));
  }
  int compression_mode = 1;  // 0 = none, 1 = online (pycall), 2 = offline (deisa-dask)
  if (!PC_status(PC_get(configs.conf_gyselax, ".CompressionBenchmark.compression_mode"))) {
    compression_mode = static_cast<int>(
        PCpp_int(configs.conf_gyselax, ".CompressionBenchmark.compression_mode"));
  }

  if (rank == 0) {
    std::cout << "Input fdistribu file name: " << fdistribu_filename
              << std::endl;
  }

  int64_t fdistribu_filename_size = fdistribu_filename.size();

  ddc::expose_to_pdi("iter_offset", iter_offset);

  PDI_multi_expose("ReadFileNames", "fdistribu_filename_size",
                   &fdistribu_filename_size, PDI_OUT, "fdistribu_filename",
                   fdistribu_filename.c_str(), PDI_OUT, NULL);

  DFieldMemSpVxVy allfequilibrium(idxrange_spvxvy_local);
  DFieldMemSpXYVxVy allfdistribu_x2D_split(idxrange_spxyvxvy_x2Dsplit);
  DFieldMemSpVxVyXY allfdistribu_v2D_split(idxrange_spvxvyxy_v2Dsplit);

  IdxRangeSpXYVxVy idxrange_spxyvxvy_v2Dsplit(idxrange_spvxvyxy_v2Dsplit);
  PDI_expose_idx_range(idxrange_spxyvxvy_v2Dsplit, "local_fdistribu");
  PDI_expose_idx_range(idxrange_glob_spxyvxvy, "global_fdistribu");

  if (nb_restart == 0) {
    init_case(idx_range_kinsp, configs, transpose, allfequilibrium,
              allfdistribu_x2D_split, allfdistribu_v2D_split);
  } else {
    DFieldMemSpXYVxVy allfdistribu_restart_output_layout(
        idxrange_spxyvxvy_v2Dsplit);

    auto allfdistribu_restart_output_layout_host =
        ddc::create_mirror_view(get_field(allfdistribu_restart_output_layout));

    ddc::PdiEvent("read_fdistribu")
        .with("fdistribu", allfdistribu_restart_output_layout_host);

    ddc::parallel_deepcopy(get_field(allfdistribu_restart_output_layout),
                           allfdistribu_restart_output_layout_host);

    transpose_layout(Kokkos::DefaultExecutionSpace(),
                     get_field(allfdistribu_v2D_split),
                     get_const_field(allfdistribu_restart_output_layout));

    if (rank == 0) {
      std::cout << "Restarted from file: " << fdistribu_filename
                << " with offset " << iter_offset << std::endl;
    }
  }

  // --> Algorithm info
  double const deltat = PCpp_double(configs.conf_gyselax, ".Algorithm.deltat");
  int const nbiter =
      static_cast<int>(PCpp_int(configs.conf_gyselax, ".Algorithm.nbiter"));

  // --> Output info
  int const nbstep_diag =
      static_cast<int>(PCpp_int(configs.conf_gyselax, ".Output.nbiter_diag"));


  // Create advection operator
  BslAdvectionSpatial<GeometryVxVyXY, SplineInterpolatorX, Real> const
      advection_x(interpolator_x);
  BslAdvectionSpatial<GeometryVxVyXY, SplineInterpolatorY, Real> const
      advection_y(interpolator_y);
  BslAdvectionVelocity<GeometryXYVxVy, SplineInterpolatorVx, Real> const
      advection_vx(interpolator_vx);
  BslAdvectionVelocity<GeometryXYVxVy, SplineInterpolatorVy, Real> const
      advection_vy(interpolator_vy);

  MpiSplitVlasovSolver const vlasov(advection_x, advection_y, advection_vx,
                                    advection_vy, transpose);

  DFieldMemVxVy const quadrature_coeffs(
      neumann_spline_quadrature_coefficients<Kokkos::DefaultExecutionSpace>(
          idxrange_vxvy, interpolator_vx.get_builder(), interpolator_vy.get_builder()));
  DFieldMemVxVy local_quadrature_coeffs(idxrange_vxvy_v2Dsplit);
  ddc::parallel_deepcopy(get_field(local_quadrature_coeffs),
                         quadrature_coeffs[idxrange_vxvy_v2Dsplit]);

  FFTPoissonSolver<IdxRangeXY> fft_poisson_solver(idxrange_xy);
  ChargeDensityCalculator const rhs_local(
      get_const_field(local_quadrature_coeffs));
  MpiChargeDensityCalculator const rhs(MPI_COMM_WORLD, rhs_local);
  QNSolver const poisson(fft_poisson_solver, rhs);

  // Create predcorr operator
  PredCorrCompress const predcorr(vlasov, poisson);

  // Starting the code
  ddc::expose_to_pdi("Nx_spline_cells",
                     ddc::discrete_space<BSplinesX>().ncells());
  ddc::expose_to_pdi("Ny_spline_cells",
                     ddc::discrete_space<BSplinesY>().ncells());
  ddc::expose_to_pdi("Nvx_spline_cells",
                     ddc::discrete_space<BSplinesVx>().ncells());
  ddc::expose_to_pdi("Nvy_spline_cells",
                     ddc::discrete_space<BSplinesVy>().ncells());
  expose_mesh_to_pdi("MeshX", idxrange_x);
  expose_mesh_to_pdi("MeshY", idxrange_y);
  expose_mesh_to_pdi("MeshVx", idxrange_vx);
  expose_mesh_to_pdi("MeshVy", idxrange_vy);
  ddc::expose_to_pdi("nbstep_diag", nbstep_diag);
  ddc::expose_to_pdi("nb_step_compression", compression_period);
  ddc::expose_to_pdi("compression_mode", compression_mode);
  ddc::expose_to_pdi("deltat", deltat);
  ddc::expose_to_pdi("Nkinspecies", idx_range_kinsp.size());
  ddc::expose_to_pdi("fdistribu_charges",
                     ddc::discrete_space<Species>().charges()[idx_range_kinsp]);
  ddc::expose_to_pdi("fdistribu_masses",
                     ddc::discrete_space<Species>().masses()[idx_range_kinsp]);

  if (rank == 0 && nb_restart == 0) {
    auto allfequilibrium_host =
        ddc::create_mirror_view_and_copy(get_field(allfequilibrium));
    ddc::PdiEvent("initial_state").with("fdistribu_eq", allfequilibrium_host);
  }

  ddc::PdiEvent("Init");

  time_points[1] = steady_clock::now();
  timing_names[0] = "Simulation initialisation";

  ddc::PdiEvent("InitBridge");

  time_points[2] = steady_clock::now();
  timing_names[1] = "Bridge initialisation";

  predcorr(get_field(allfdistribu_v2D_split), deltat, nbiter);

  ddc::PdiEvent("EndSimulation").with("iter", nbiter);

  time_points[3] = steady_clock::now();
  timing_names[2] = "full simulation";

  if (rank == 0) {
    double durations[4];
    for (int i = 1; i <= 3; i++) {
      durations[i-1] =
          std::chrono::duration<double>(time_points[i] - time_points[i-1])
              .count();
    }
    durations[3] =
        std::chrono::duration<double>(time_points[3] - time_points[0]).count();
    timing_names[3] = "total";
    for (int i = 0; i < 4; i++) {
      cout << "Time " << timing_names[i] << ": " << durations[i] << "s" << endl;
    }
    // Use the new function to write timing stats as a table
    //write_cpu_time_stats(rank, durations, timing_names, timing_names.size());
  }


  PC_tree_destroy(&configs.conf_pdi);

  PDI_finalize();

  MPI_Finalize();

  PC_tree_destroy(&configs.conf_gyselax);

  return EXIT_SUCCESS;
}

// SPDX-License-Identifier: MIT
#include <chrono>
#include <cmath>
#include <cstdlib>
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
#include "predcorr.hpp"
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

  SplineXBuilder const builder_x(idxrange_x);
  SplineYBuilder const builder_y(idxrange_y);
  SplineVxBuilder const builder_vx(idxrange_vx);
  SplineVyBuilder const builder_vy(idxrange_vy);

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

  if (nb_restart == 0) {
    MaxwellianEquilibrium const init_fequilibrium =
        MaxwellianEquilibrium::init_from_input(idx_range_kinsp,
                                               configs.conf_gyselax);
    init_fequilibrium(get_field(allfequilibrium));

    SingleModePerturbInitialisation const init =
        SingleModePerturbInitialisation::init_from_input(
            get_const_field(allfequilibrium), idx_range_kinsp,
            configs.conf_gyselax);
    init(get_field(allfdistribu_x2D_split));

    transpose(Kokkos::DefaultExecutionSpace(),
              get_field(allfdistribu_v2D_split),
              get_const_field(allfdistribu_x2D_split));
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
  double const time_diag =
      PCpp_double(configs.conf_gyselax, ".Output.time_diag");
  int const nbstep_diag = int(time_diag / deltat);

  // Create spline evaluator
  ddc::PeriodicExtrapolationRule<X> bv_x_min;
  ddc::PeriodicExtrapolationRule<X> bv_x_max;
  SplineXEvaluator const spline_x_evaluator(bv_x_min, bv_x_max);

  ddc::PeriodicExtrapolationRule<Y> bv_y_min;
  ddc::PeriodicExtrapolationRule<Y> bv_y_max;
  SplineYEvaluator const spline_y_evaluator(bv_y_min, bv_y_max);

  ddc::ConstantExtrapolationRule<Vx> bv_vx_min(
      ddc::coordinate(idxrange_vx.front()));
  ddc::ConstantExtrapolationRule<Vx> bv_vx_max(
      ddc::coordinate(idxrange_vx.back()));
  SplineVxEvaluator const spline_vx_evaluator(bv_vx_min, bv_vx_max);

  ddc::ConstantExtrapolationRule<Vy> bv_vy_min(
      ddc::coordinate(idxrange_vy.front()));
  ddc::ConstantExtrapolationRule<Vy> bv_vy_max(
      ddc::coordinate(idxrange_vy.back()));
  SplineVyEvaluator const spline_vy_evaluator(bv_vy_min, bv_vy_max);

  // Create advection operator
  BslAdvectionSpatial<GeometryVxVyXY, GridX, SplineXBuilder,
                      SplineXEvaluator> const advection_x(builder_x,
                                                          spline_x_evaluator);
  BslAdvectionSpatial<GeometryVxVyXY, GridY, SplineYBuilder,
                      SplineYEvaluator> const advection_y(builder_y,
                                                          spline_y_evaluator);
  BslAdvectionVelocity<GeometryXYVxVy, GridVx, SplineVxBuilder,
                       SplineVxEvaluator> const
      advection_vx(builder_vx, spline_vx_evaluator);
  BslAdvectionVelocity<GeometryXYVxVy, GridVy, SplineVyBuilder,
                       SplineVyEvaluator> const
      advection_vy(builder_vy, spline_vy_evaluator);

  MpiSplitVlasovSolver const vlasov(advection_x, advection_y, advection_vx,
                                    advection_vy, transpose);

  DFieldMemVxVy const quadrature_coeffs(
      neumann_spline_quadrature_coefficients<Kokkos::DefaultExecutionSpace>(
          idxrange_vxvy, builder_vx, builder_vy));
  DFieldMemVxVy local_quadrature_coeffs(idxrange_vxvy_v2Dsplit);
  ddc::parallel_deepcopy(get_field(local_quadrature_coeffs),
                         quadrature_coeffs[idxrange_vxvy_v2Dsplit]);

  FFTPoissonSolver<IdxRangeXY> fft_poisson_solver(idxrange_xy);
  ChargeDensityCalculator const rhs_local(
      get_const_field(local_quadrature_coeffs));
  MpiChargeDensityCalculator const rhs(MPI_COMM_WORLD, rhs_local);
  QNSolver const poisson(fft_poisson_solver, rhs);

  // Create predcorr operator
  PredCorr const predcorr(vlasov, poisson);

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

  steady_clock::time_point const start = steady_clock::now();

  predcorr(get_field(allfdistribu_v2D_split), deltat, nbiter);

  steady_clock::time_point const end = steady_clock::now();

  double const simulation_time =
      std::chrono::duration<double>(end - start).count();
  std::cout << "Simulation time: " << simulation_time << "s\n";

  PC_tree_destroy(&configs.conf_pdi);

  PDI_finalize();

  MPI_Finalize();

  PC_tree_destroy(&configs.conf_gyselax);

  return EXIT_SUCCESS;
}

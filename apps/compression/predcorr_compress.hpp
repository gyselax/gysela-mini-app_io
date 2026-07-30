// SPDX-License-Identifier: MIT

#pragma once

#include "geometry_xyvxvy.hpp"
#include "itimesolver.hpp"

class IQNSolver;
class IVlasovSolver;

/**
 * @brief Predictor-corrector time solver for the compression benchmark app.
 *
 * Identical to gyselalibxx's `PredCorr` (geometryXYVxVy/time_integration),
 * except that it copies the host-side `fdistribu` buffer back into the
 * device-resident field after the "iteration"/"last_iteration" PDI events.
 *
 * `PredCorr` only ever copies the device field to the host to expose it to
 * PDI for diagnostics/output; the in-situ compression handlers wired up in
 * pdi_out_diags.yaml mutate that host buffer in place (`fdistribu[...] =
 * approx`), but with no code to write it back, that mutation was silently
 * discarded and the predictor/corrector kept advancing the original,
 * uncompressed device field. This class adds the missing copy-back so
 * in-situ compression actually affects the simulation trajectory.
 */
class PredCorrCompress : public ITimeSolver
{
private:
    IVlasovSolver const& m_vlasov_solver;

    IQNSolver const& m_poisson_solver;

public:
    /**
     * @brief Creates an instance of the predictor-corrector class.
     * @param[in] vlasov_solver A solver for a Boltzmann equation.
     * @param[in] poisson_solver A solver for a Poisson equation.
     */
    PredCorrCompress(IVlasovSolver const& vlasov_solver, IQNSolver const& poisson_solver);

    ~PredCorrCompress() override = default;

    /**
     * @brief Solves the Vlasov-Poisson system.
     * @param[in, out] allfdistribu On input : the initial value of the distribution function.
     *                              On output : the value of the distribution function after solving
     *                              the Vlasov-Poisson system a given number of iterations.
     * @param[in] dt The timestep.
     * @param[in] steps The number of iterations to be performed by the predictor-corrector.
     * @return The distribution function after solving the system.
     */
    DFieldSpVxVyXY operator()(DFieldSpVxVyXY allfdistribu, Real dt, int steps = 1) const override;
};

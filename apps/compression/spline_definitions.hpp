// SPDX-License-Identifier: MIT

#pragma once

#include <spline_interpolation.hpp>
#include <geometry_xyvxvy.hpp>

struct BSplinesX : ddc::UniformBSplines<X, 3> {};
struct BSplinesVx : ddc::UniformBSplines<Vx, 3> {};

struct BSplinesY : ddc::UniformBSplines<Y, 3> {};
struct BSplinesVy : ddc::UniformBSplines<Vy, 3> {};

auto constexpr SplineXClosure = ddc::SplineBuilderClosure::PERIODIC;
auto constexpr SplineVxClosure = ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE;

using SplineInterpPointsX =
    ddc::GrevilleInterpolationPoints<BSplinesX, SplineXClosure, SplineXClosure>;
using SplineInterpPointsVx =
    ddc::GrevilleInterpolationPoints<BSplinesVx, SplineVxClosure,
                                     SplineVxClosure>;

ddc::SplineBuilderClosure constexpr SplineYClosure =
    ddc::SplineBuilderClosure::PERIODIC;
ddc::SplineBuilderClosure constexpr SplineVyClosure =
    ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE;

// IDim initialisers
using SplineInterpPointsY =
    ddc::GrevilleInterpolationPoints<BSplinesY, SplineYClosure, SplineYClosure>;
using SplineInterpPointsVy =
    ddc::GrevilleInterpolationPoints<BSplinesVy, SplineVyClosure,
                                     SplineVyClosure>;

using SplineInterpolatorX =
    SplineInterpolator<Kokkos::DefaultExecutionSpace, IdxRange<BSplinesX>,
                       IdxRange<GridX>, ExtrapolationRule::Periodic,
                       SplineBoundaryClosures<SplineXClosure, SplineXClosure>>;

using SplineInterpolatorVx = SplineInterpolator<
    Kokkos::DefaultExecutionSpace, IdxRange<BSplinesVx>, IdxRange<GridVx>,
    ExtrapolationRule::Constant_Constant,
    SplineBoundaryClosures<SplineVxClosure, SplineVxClosure>>;

// SplineBuilder and SplineEvaluator definition
using SplineInterpolatorY =
    SplineInterpolator<Kokkos::DefaultExecutionSpace, IdxRange<BSplinesY>,
                       IdxRange<GridY>, ExtrapolationRule::Periodic,
                       SplineBoundaryClosures<SplineYClosure, SplineYClosure>>;

using SplineInterpolatorVy = SplineInterpolator<
    Kokkos::DefaultExecutionSpace, IdxRange<BSplinesVy>, IdxRange<GridVy>,
    ExtrapolationRule::Constant_Constant,
    SplineBoundaryClosures<SplineVyClosure, SplineVyClosure>>;

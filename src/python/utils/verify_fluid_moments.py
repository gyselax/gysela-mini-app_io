import argparse
import xarray as xr
from fluid_moments import FluidMoments

# -------------------------------------------------
# 3. Execution Script (Dask-Backed)
# -------------------------------------------------

def verify_fluid_moments(
    f_path="fdistribu_5D_output.h5",
    moments_path="fluid_moments.h5",
    output_path="fluid_moments.nc",
):
    ds_f = xr.open_dataset(f_path, chunks={"phony_dim_1": 1, "phony_dim_2": -1})
    ds_ref = xr.open_dataset(moments_path, chunks="auto")

    rename_dict = {
        "phony_dim_0": "species",
        "phony_dim_1": "tor1",
        "phony_dim_2": "tor2",
        "phony_dim_3": "tor3",
        "phony_dim_4": "vpar",
        "phony_dim_5": "mu",
    }

    ds_f = ds_f.rename({k: v for k, v in rename_dict.items() if k in ds_f.dims})
    ds_ref = ds_ref.rename({k: v for k, v in rename_dict.items() if k in ds_ref.dims})

    moments_calc = FluidMoments(ds_f["vpar"], ds_f["mu"])

    fdistribu = ds_f["fdistribu_sptor3Dv2D"]

    density = moments_calc.compute_density(fdistribu)
    mean_velocity = moments_calc.compute_velocity(fdistribu, density)
    temperature = moments_calc.compute_temperature(
        fdistribu,
        density,
        mean_velocity,
    )

    ds_out = xr.Dataset(
        {
            "density": density,
            "mean_velocity": mean_velocity,
            "temperature": temperature,
        }
    ).compute()

    diff = abs(ds_out - ds_ref).to_array()
    print(f"Max Abs Error: {diff.max().values}")

    ds_out.to_netcdf(output_path)
    print(f"Wrote output moments to: {output_path}")


def setup_parser(parser):
    parser.add_argument(
        "--fdistribu",
        default="fdistribu_5D_output.h5",
        help="Input distribution HDF5 file.",
    )
    parser.add_argument(
        "--moments",
        default="fluid_moments.h5",
        help="Reference fluid moments HDF5 file.",
    )
    parser.add_argument(
        "--output",
        default="fluid_moments.nc",
        help="Output NetCDF file.",
    )


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        setup_parser(parser)
        args = parser.parse_args()

    verify_fluid_moments(
        f_path=args.fdistribu,
        moments_path=args.moments,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

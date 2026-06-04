"""Command-line interface for GYSELA mini-app Python utilities."""

import argparse

from utils.read_timing_stats import setup_parser as setup_read_timing_parser
from utils.read_timing_stats import main as read_timing_main
from utils.verify_fluid_moments import setup_parser as setup_verify_moments_parser
from utils.verify_fluid_moments import main as verify_moments_main


def main():
    parser = argparse.ArgumentParser(
        description="GYSELA mini-app Python tools",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Available commands",
        required=False,
    )

    read_timing_description = "Read and display CPU timing statistics from an HDF5 file"
    read_timing_parser = subparsers.add_parser(
        "read-timing-stats",
        description=read_timing_description,
        help=read_timing_description,
    )
    setup_read_timing_parser(read_timing_parser)
    read_timing_parser.set_defaults(func=read_timing_main)

    verify_moments_description = "Verify fluid moments against a reference HDF5 file"
    verify_moments_parser = subparsers.add_parser(
        "verify-fluid-moments",
        description=verify_moments_description,
        help=verify_moments_description,
    )
    setup_verify_moments_parser(verify_moments_parser)
    verify_moments_parser.set_defaults(func=verify_moments_main)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read and display CPU timing statistics from HDF5 file."""
import sys
import h5py
import numpy as np

def read_timing_stats(filename):
    """Read timing statistics from HDF5 file."""
    try:
        with h5py.File(filename, 'r') as f:
            print("CPU Timing Statistics")
            print("=" * 50)
            
            # Read timing names (2D char array)
            if 'timing_names' in f:
                names_ds = f['timing_names']
                names = []
                # Convert 2D char array to list of strings
                for i in range(names_ds.shape[0]):
                    name_bytes = names_ds[i].tobytes()
                    name_str = name_bytes.decode('utf-8').rstrip('\x00')
                    names.append(name_str)
                
                # Read timing values
                if 'timing_values' in f:
                    values = f['timing_values'][:]
                    
                    # Display as table
                    print(f"{'Name':<20} {'Time (s)':>15}")
                    print("-" * 50)
                    for name, value in zip(names, values):
                        print(f"{name:<20} {value:>15.6f}")
                    print("-" * 50)
                else:
                    print("Warning: timing_values not found")
                    print("Names found:")
                    for i, name in enumerate(names):
                        print(f"  {i}: {name}")
            else:
                print("Error: timing_names dataset not found")
                print("Available datasets:", list(f.keys()))
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "cpu_time_stats.h5"
    read_timing_stats(filename)


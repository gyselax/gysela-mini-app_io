import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

from compression_methods.neural_network import NeuralNetworkCompressor

def generate_synthetic_fdistribu(n_species, nx, ny, nvx, nvy):
    """Generates a synthetic 5D distribution function: a maxwellian distribution with a slight periodic spatial perturbation."""
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)
    vx = np.linspace(-6.0, 6.0, nvx, endpoint=True)
    vy = np.linspace(-6.0, 6.0, nvy, endpoint=True)
    
    Xg, Yg, VXg, VYg = np.meshgrid(x, y, vx, vy, indexing="ij")
    
    ## f(x, y, vx, vy) = exp(-(vx^2 + vy^2)/2) * (1 + 0.1 * cos(x) * sin(y))
    f_base = np.exp(-0.5 * (VXg**2 + VYg**2)) * (1.0 + 0.1 * np.cos(Xg) * np.sin(Yg))
    
    fdistribu = np.zeros((n_species, nx, ny, nvx, nvy), dtype=np.float64)
    for isp in range(n_species):
        fdistribu[isp] = f_base * (isp + 1)  #Basic variation between species
        
    return fdistribu

def test_inr_plumbing():
    """
    Verifies that the compressor correctly ingests, optimizes, and spits out a 5D tensor. 
    Uses very few iterations so that the test is instantaneous.
    """
    n_species = 2
    nx,ny = 8, 8
    nvx,nvy = 8, 8
    
    print("\n---Synthetic data generation---")
    f_original = generate_synthetic_fdistribu(n_species, nx, ny, nvx, nvy)
    print(f"Shape of fdistribu: {f_original.shape}")
    
    compressor = NeuralNetworkCompressor(
        x_min=0.0, x_max=2*np.pi,
        y_min=0.0, y_max=2*np.pi,
        vx_min=-6.0, vx_max=6.0,
        vy_min=-6.0, vy_max=6.0,
        arch="periodic_siren_deep_128",
        lr=1e-3,
        max_iters=15,
        batch_size=512,
        lbfgs_iters=5,
        verbose=True
    )
    
    print("\n-- Start of Compression / Decompression ---")
    compressed_dict, f_reconstructed = compressor.compress_decompress_array(f_original)
    
    print("\n---Checking---")
    #shape verification
    assert f_reconstructed.shape == f_original.shape, \
        f"Dimensionality error: expected {f_original.shape}, got {f_reconstructed.shape}"
    print("[OK] The reconstruction dimensions are correct.")
    
    #verification of model creation by species
    assert len(compressed_dict["models"]) == n_species, \
        f"Model error : expected {n_species} models, got {len(compressed_dict['models'])}"
    print("[OK] The correct number of models has been generated.")
    
    #verification of the decreasing of th loss 
    metrics = compressor.get_extra_metrics()
    final_losses = metrics["final_loss_per_species"]
    assert final_losses is not None and len(final_losses) == n_species, "The losses were not properly recorded."
    print(f"[OK] Final losses per species : {final_losses}")
    
    #Payload save/load test
    test_payload_path = "test_payload.npz"
    try:
        compressor.save_compressed_payload(test_payload_path, compressed_dict)
        assert os.path.exists(test_payload_path), "Payload file was not created."
        print(f"[OK] Payload saved successfully ({os.path.getsize(test_payload_path) / 1024:.2f} KB).")
    finally:
        #cleaning
        if os.path.exists(test_payload_path):
            os.remove(test_payload_path)
            
if __name__ == "__main__":
    test_inr_plumbing()
    print("\n All INR unit tests have passed successfully!")
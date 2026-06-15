#include <pdi.h>
#include <ddc/ddc.hpp>


extern "C" {
void fm_mem_copy(void)
{
    using View6D = Kokkos::View<double******, Kokkos::DefaultExecutionSpace>;
    using UnmanagedView6D = Kokkos::View<double******, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    
    size_t* extents;
    PDI_access("local_fdistribu_extents", (void**)&extents, PDI_IN);
    
    void* data_d; 
    PDI_access("local_fdistribu_sptor3Dv2D_d", &data_d, PDI_IN);
    
    double* raw_data = static_cast<double*>(data_d);
    UnmanagedView6D wrapped_view(raw_data, extents[0], extents[1], extents[2], extents[3], extents[4], extents[5]);
    View6D destination_view("dest_view_6d", extents[0], extents[1], extents[2], extents[3], extents[4], extents[5]);

    Kokkos::deep_copy(destination_view, wrapped_view);

    PDI_multi_expose("FluidMoments", 
                     "local_fdistribu_sptor3Dv2D", destination_view.data(), PDI_OUT,
                     NULL);
    
    PDI_release("local_fdistribu_sptor3Dv2D_d");
    PDI_release("local_fdistribu_extents");
}

void fdistribu_mem_copy(void)
{
    using View6D = Kokkos::View<double******, Kokkos::DefaultExecutionSpace>;
    using UnmanagedView6D = Kokkos::View<double******, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    
    size_t* extents;
    PDI_access("local_fdistribu_extents", (void**)&extents, PDI_IN);
    
    void* data_d; 
    PDI_access("local_fdistribu_sptor3Dv2D_d", &data_d, PDI_IN);
    
    double* raw_data = static_cast<double*>(data_d);
    UnmanagedView6D wrapped_view(raw_data, extents[0], extents[1], extents[2], extents[3], extents[4], extents[5]);
    View6D destination_view("dest_view_6d", extents[0], extents[1], extents[2], extents[3], extents[4], extents[5]);

    Kokkos::deep_copy(destination_view, wrapped_view);

    PDI_multi_expose("write_fdistribu", 
                     "local_fdistribu_sptor3Dv2D", destination_view.data(), PDI_OUT,
                     NULL);
    
    PDI_release("local_fdistribu_sptor3Dv2D_d");
    PDI_release("local_fdistribu_extents");
}

}
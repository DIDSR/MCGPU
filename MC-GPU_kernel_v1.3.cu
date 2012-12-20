
////////////////////////////////////////////////////////////////////////////////
//
//              ****************************
//              *** MC-GPU , version 1.3 ***
//              ****************************
//                                          
//!  Definition of the CUDA GPU kernel for the simulation of x ray tracks in a voxelized geometry.
//!  This kernel has been optimized to yield a good performance in the GPU but can still be
//!  compiled in the CPU without problems. All the CUDA especific commands are enclosed in
//!  pre-processor directives that are skipped if the parameter "USING_CUDA" is not defined
//!  at compilation time.
//
//        ** DISCLAIMER **
//
// This software and documentation (the "Software") were developed at the Food and
// Drug Administration (FDA) by employees of the Federal Government in the course
// of their official duties. Pursuant to Title 17, Section 105 of the United States
// Code, this work is not subject to copyright protection and is in the public
// domain. Permission is hereby granted, free of charge, to any person obtaining a
// copy of the Software, to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, or sell copies of the Software or derivatives, and to permit persons
// to whom the Software is furnished to do so. FDA assumes no responsibility
// whatsoever for use by other parties of the Software, its source code,
// documentation or compiled executables, and makes no guarantees, expressed or
// implied, about its quality, reliability, or any other characteristic. Further,
// use of this code in no way implies endorsement by the FDA or confers any
// advantage in regulatory decisions.  Although this software can be redistributed
// and/or modified freely, we ask that any derivative works bear some notice that
// they are derived from it, and any modified versions bear some notice that they
// have been modified.
//                                                                            
//
//!                     @file    MC-GPU_kernel_v1.3.cu
//!                     @author  Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)
//!                     @date    2012/12/12
//                       -- Original code started on:  2009/04/14
//
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//!  Initialize the image array, ie, set all pixels to zero
//!  Essentially, this function has the same effect as the command: 
//!   "cutilSafeCall(cudaMemcpy(image_device, image, image_bytes, cudaMemcpyHostToDevice))";
//!  
//!  CUDA performs some initialization work the first time a GPU kernel is called.
//!  Therefore, calling a short kernel before the real particle tracking is performed
//!  may improve the accuracy of the timing measurements in the relevant kernel.
//!  
//!       @param[in,out] image   Pointer to the image array.
//!       @param[in] pixels_per_image  Number of pixels in the image (ie, elements in the array).
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__global__
void init_image_array_GPU(unsigned long long int* image, int pixels_per_image)
{
  int my_pixel = threadIdx.x + blockIdx.x*blockDim.x;
  if (my_pixel < pixels_per_image)
  {
    // -- Set the current pixel to 0 and return, avoiding overflow when more threads than pixels are used:
    image[my_pixel] = (unsigned long long int)(0);    // Initialize non-scatter image
    my_pixel += pixels_per_image;                     //  (advance to next image)
    image[my_pixel] = (unsigned long long int)(0);    // Initialize Compton image
    my_pixel += pixels_per_image;                     //  (advance to next image)
    image[my_pixel] = (unsigned long long int)(0);    // Initialize Rayleigh image
    my_pixel += pixels_per_image;                     //  (advance to next image)
    image[my_pixel] = (unsigned long long int)(0);    // Initialize multi-scatter image
  }
}

// ////////////////////////////////////////////////////////////////////////////////
// //!  Initialize the dose deposition array, ie, set all voxel doses to zero
// //!  
// //!       @param[in,out] dose   Pointer to the dose mean and sigma arrays.
// //!       @param[in] num_voxels_dose  Number of voxels in the dose ROI (ie, elements in the arrays).
// ////////////////////////////////////////////////////////////////////////////////
// __global__
// void init_dose_array_GPU(ulonglong2* voxels_Edep, int num_voxels_dose)
// {  
//   int my_voxel = threadIdx.x + blockIdx.x*blockDim.x;
//   register ulonglong2 ulonglong2_zero;
//   ulonglong2_zero.x = ulonglong2_zero.y = (unsigned long long int) 0;
//   if (my_voxel < num_voxels_dose)
//   {
//     dose[my_voxel] = ulonglong2_zero;    // Set the current voxel to (0,0) and return, avoiding overflow
//   }
// }

#endif

 
////////////////////////////////////////////////////////////////////////////////
//!  Main function to simulate x-ray tracks inside a voxelized geometry.
//!  Secondary electrons are not simulated (in photoelectric and Compton 
//!  events the energy is locally deposited).
//!
//!  The following global variables, in  the GPU __constant__ memory are used:
//!           voxel_data_CONST, 
//!           source_energy_data_CONST,
//!           detector_data_CONST, 
//!           mfp_table_data_CONST.
//!
//!       @param[in] history_batch  Particle batch number (only used in the CPU version when CUDA is disabled!, the GPU uses the built-in variable threadIdx)
//!       @param[in] num_p  Projection number in the CT simulation. This variable defines a specific angle and the corresponding source and detector will be used.
//!       @param[in] histories_per_thread   Number of histories to simulate for each call to this function (ie, for GPU thread).
//!       @param[in] seed_input   Random number generator seed (the same seed is used to initialize the two MLCGs of RANECU).
//!       @param[in] voxel_mat_dens   Pointer to the voxel densities and material vector (the voxelized geometry), stored in GPU glbal memory.
//!       @param[in] mfp_Woodcock_table    Two parameter table for the linear interpolation of the Woodcock mean free path (MFP) (stored in GPU global memory).
//!       @param[in] mfp_table_a   First element for the linear interpolation of the interaction mean free paths (stored in GPU global memory).
//!       @param[in] mfp_table_b   Second element for the linear interpolation of the interaction mean free paths (stored in GPU global memory).
//!       @param[in] rayleigh_table   Pointer to the table with the data required by the Rayleigh interaction sampling, stored in GPU global memory.
//!       @param[in] compton_table   Pointer to the table with the data required by the Compton interaction sampling, stored in GPU global memory.
//!       @param[in,out] image   Pointer to the image vector in the GPU global memory.
//!       @param[in,out] dose   Pointer to the array containing the 3D voxel dose (and its uncertainty) in the GPU global memory.
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__global__ void track_particles(int histories_per_thread,
                                int num_p,      // For a CT simulation: allocate space for up to MAX_NUM_PROJECTIONS projections.
                                int seed_input,
                                unsigned long long int* image,
                                ulonglong2* voxels_Edep,
                                float2* voxel_mat_dens,
                                float2* mfp_Woodcock_table,
                                float3* mfp_table_a,
                                float3* mfp_table_b,
                                struct rayleigh_struct* rayleigh_table,
                                struct compton_struct* compton_table,
                                struct detector_struct* detector_data_array,
                                struct source_struct* source_data_array, 
                                ulonglong2* materials_dose)
#else
           void track_particles(int history_batch,             // This variable is not required in the GPU, it uses the thread ID           
                                int histories_per_thread,
                                int num_p,
                                int seed_input,
                                unsigned long long int* image,
                                ulonglong2* voxels_Edep,
                                float2* voxel_mat_dens,
                                float2* mfp_Woodcock_table,
                                float3* mfp_table_a,
                                float3* mfp_table_b,
                                struct rayleigh_struct* rayleigh_table,
                                struct compton_struct* compton_table,
                                struct detector_struct* detector_data_array,
                                struct source_struct* source_data_array, 
                                ulonglong2* materials_dose)
#endif
{
  // -- Declare the track state variables:
  float3 position, direction;
  float energy, step, prob, randno, mfp_density, mfp_Woodcock;
  float3 mfp_table_read_a, mfp_table_read_b;
  int2 seed;
  int index;
  int material0,        // Current material, starting at 0 for 1st material
      material_old;     // Flag to mark a material or energy change
  signed char scatter_state;    // Flag for scatter images: scatter_state=0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter.

  // -- Store the Compton table in shared memory from global memory:
  //    For Compton and Rayleigh the access to memory is not coherent and the caching capability do not speeds up the accesses, they actually slows down the acces to other data.
#ifdef USING_CUDA
  __shared__
#endif
  struct compton_struct cgco_SHARED;  
#ifdef USING_CUDA
  __shared__
#endif
  struct detector_struct detector_data_SHARED;
#ifdef USING_CUDA
  __shared__
#endif 
  struct source_struct source_data_SHARED;    

    
#ifdef USING_CUDA
  if (0==threadIdx.x)  // First GPU thread copies the variables to shared memory
  {
#endif

    // -Copy the current source, detector data from global to shared memory for fast access:
    source_data_SHARED    = source_data_array[num_p];      
    detector_data_SHARED  = detector_data_array[num_p];    // Copy the long array to a single instance in shared memory for the current projection
        
    // -Copy the compton data to shared memory:
    cgco_SHARED = *compton_table;
    
#ifdef USING_CUDA
  }
  __syncthreads();     // Make sure all threads will see the initialized shared variable  
#endif


  // -- Initialize the RANECU generator in a position far away from the previous history:
#ifdef USING_CUDA
  init_PRNG((threadIdx.x + blockIdx.x*blockDim.x), histories_per_thread, seed_input, &seed);   // Using a 1D block
#else
  init_PRNG(history_batch, histories_per_thread, seed_input, &seed);
#endif

  
  // -- Loop for the "histories_per_thread" particles in the current history_batch:

  for( ; histories_per_thread>0; histories_per_thread--)
  {
        //  printf("\n\n********* NEW HISTORY:  %d    [seeds: %d, %d]\n\n", histories_per_thread, seed.x, seed.y); //  fflush(stdout);  // !!Verbose!! calling printf from the GPU is possible but if multiple threads call it at the same time some output will be lost.

    int absvox = 1;
    
    // -- Call the source function to get a primary x ray:
    source(&position, &direction, &energy, &seed, &absvox, &source_data_SHARED, &detector_data_SHARED);

    scatter_state = (signed char)0;     // Reset previous scatter state: new non-scattered particle loaded

    // -- Find the current energy bin by truncation (this could be pre-calculated for a monoenergetic beam):    
    //    The initialization host code made sure that the sampled energy will always be within the tabulated energies (index never negative or too large).
#ifdef USING_CUDA
    index = __float2int_rd((energy-mfp_table_data_CONST.e0)*mfp_table_data_CONST.ide);  // Using CUDA function to convert float to integer rounding down (towards minus infinite)
#else
    index = (int)((energy-mfp_table_data_CONST.e0)*mfp_table_data_CONST.ide + 0.00001f);    // Adding EPSILON to truncate to INT towards minus infinite. There may be a small error for energy<=mfp_table_data_CONST.e0 but this case is irrelevant (particles will always have more energy than e0).
#endif          

  
    // -- Get the minimum mfp at the current energy using linear interpolation (Woodcock tracking):      
    {
      float2 mfp_Woodcock_read = mfp_Woodcock_table[index];   // Read the 2 parameters for the linear interpolation in a single read from global memory
      mfp_Woodcock = mfp_Woodcock_read.x + energy * mfp_Woodcock_read.y;   // Interpolated minimum MFP          
    }


    // -- Reset previous material to force a recalculation of the MFPs (negative materials are not allowed in the voxels):
    material_old  = -1;

    // *** X-ray interaction loop:
    for(;;)
    {
      
      if (absvox<0)   // !!DeBuG!!  MC-GPU_v1.3 ==> if I move this "if" above the code runs much slower!? Why???
          break;    // -- Primary particle was not pointing to the voxel region! (but may still be detected after moving in vacuum in a straight line).      


      // *** Virtual interaction loop:  // New loop structure in MC-GPU_v1.3: simulate all virtual events before sampling Compton & Rayleigh:  // !!DeBuG!!
      
      float2 matdens;
      short3 voxel_coord;    // Variable used only by DOSE TALLY

      do
      {     
        step = -(mfp_Woodcock)*logf(ranecu(&seed));   // Using the minimum MFP in the geometry for the input energy (Woodcock tracking)
          
        position.x += step*direction.x;
        position.y += step*direction.y;
        position.z += step*direction.z;

        // -- Locate the new particle in the voxel geometry:      
        absvox = locate_voxel(&position, &voxel_coord);   // Get the voxel number at the current position and the voxel coordinates (used to check if inside the dose ROI in DOSE TALLY).
        if (absvox<0)
          break;    // -- Particle escaped the voxel region! ("index" is still >0 at this moment)
          
        matdens = voxel_mat_dens[absvox];     // Get the voxel material and density in a single read from global memory
        material0 = (int)(matdens.x - 1);   // Set the current material by truncation, and set 1st material to value '0'.

        // -- Get the data for the linear interpolation of the interaction MFPs, in case the energy or material have changed:
        if (material0 != material_old)
        {
          mfp_table_read_a = mfp_table_a[index*(MAX_MATERIALS)+material0];
          mfp_table_read_b = mfp_table_b[index*(MAX_MATERIALS)+material0];
          material_old = material0;                                              // Store the new material
        }
        
        // *** Apply Woodcock tracking:
        mfp_density = mfp_Woodcock * matdens.y;
        // -- Calculate probability of delta scattering, using the total mean free path for the current material and energy (linear interpolation):
        prob = 1.0f - mfp_density * (mfp_table_read_a.x + energy * mfp_table_read_b.x);
        randno = ranecu(&seed);    // Sample uniform PRN
      }
      while (randno<prob);   // [Iterate if there is a delta scattering event]

      if (absvox<0)
        break;    // -- Particle escaped the voxel region! Break the interaction loop to call tally image.

        
      // The GPU threads will be stopped and waiting here until ALL threads have a REAL event: 

      // -- Real event takes place! Check the kind of event and sample the effects of the interaction:
      
      prob += mfp_density * (mfp_table_read_a.y + energy * mfp_table_read_b.y);    // Interpolate total Compton MFP ('y' component)
      if (randno<prob)   // [Checking Compton scattering]
      {
        // *** Compton interaction:

        //  -- Sample new direction and energy:
        double costh_Compton;
        randno = energy;     // Save temporal copy of the particle energy (variable randno not necessary until next sampling). DOSE TALLY
        
        GCOa(&energy, &costh_Compton, &material0, &seed, &cgco_SHARED);
        rotate_double(&direction, costh_Compton, /*phi=2*pi*PRN=*/ 6.28318530717958647693*ranecu_double(&seed));

        randno = energy - randno;   // Save temporal copy of the negative of the energy lost in the interaction.  DOSE TALLY

        // -- Find the new energy interval:
#ifdef USING_CUDA
        index = __float2int_rd((energy-mfp_table_data_CONST.e0)*mfp_table_data_CONST.ide);  // Using CUDA function to convert float to integer rounding down (towards minus infinite)
#else
        index = (int)((energy-mfp_table_data_CONST.e0)*mfp_table_data_CONST.ide + 0.00001f);    // Adding EPSILON to truncate to INT
#endif          

        
        if (index>-1)  // 'index' will be negative only when the energy is below the tabulated minimum energy: particle will be then absorbed (rejected) after tallying the dose.
        {          
          // -- Get the Woodcock MFP for the new energy (energy above minimum cutoff):
          float2 mfp_Woodcock_read = mfp_Woodcock_table[index];   // Read the 2 parameters for the linear interpolation in a single read from global memory
          mfp_Woodcock = mfp_Woodcock_read.x + energy * mfp_Woodcock_read.y;   // Interpolated minimum MFP

          material_old = -2;    // Set an impossible material to force an update of the MFPs data for the nex energy interval

          // -- Update scatter state:
          if (scatter_state==(signed char)0)
            scatter_state = (signed char)1;   // Set scatter_state == 1: Compton scattered particle
          else
            scatter_state = (signed char)3;   // Set scatter_state == 3: Multi-scattered particle
        }

      }
      else
      {
        prob += mfp_density * (mfp_table_read_a.z + energy * mfp_table_read_b.z);    // Interpolate total Rayleigh MFP ('z' component)
        if (randno<prob)   // [Checking Rayleigh scattering]
        {
          // *** Rayleigh interaction:

          //  -- Sample angular deflection:
          double costh_Rayleigh;
          float pmax_current = rayleigh_table->pmax[(index+1)*MAX_MATERIALS+material0];   // Get max (ie, value for next bin?) cumul prob square form factor for Rayleigh sampling

          GRAa(&energy, &costh_Rayleigh, &material0, &pmax_current, &seed, rayleigh_table);
          rotate_double(&direction, costh_Rayleigh, /*phi=2*pi*PRN=*/ 6.28318530717958647693*ranecu_double(&seed));

          // -- Update scatter state:
          if (scatter_state==(signed char)0)
            scatter_state = (signed char)2;   // Set scatter_state == 1: Rayleigh scattered particle
          else
            scatter_state = (signed char)3;   // Set scatter_state == 3: Multi-scattered particle

        }
        else
        {
          // *** Photoelectric interaction (or pair production): mark particle for absorption after dose tally (ie, index<0)!
          randno = -energy;   // Save temporal copy of the (negative) energy deposited in the interaction (variable randno not necessary anymore).
          index = -11;       // A negative "index" marks that the particle was absorved and that it will never arrive at the detector.
        }
      }
    
      //  -- Tally the dose deposited in Compton and photoelectric interactions:
      if (randno<-0.001f)
      {
        float Edep = -1.0f*randno;   // If any energy was deposited, this variable will temporarily store the negative value of Edep.
        
        //  -- Tally the dose deposited in the current material, if enabled (ie, array allocated and not null):
        if (materials_dose!=NULL)
          tally_materials_dose(&Edep, &material0, materials_dose);    // !!tally_materials_dose!!

        //  -- Tally the energy deposited in the current voxel, if enabled (tally disabled when dose_ROI_x_max_CONST is negative). DOSE TALLY
        if (dose_ROI_x_max_CONST > -1)
          tally_voxel_energy_deposition(&Edep, &voxel_coord, voxels_Edep);

      }    

      // -- Break interaction loop for particles that have been absorved or with energy below the tabulated cutoff: particle is "absorbed" (ie, track discontinued).
      if (index<0)
        break;  
      
    }   // [Cycle the X-ray interaction loop]

    if (index>-1)
    {
      // -- Particle escaped the voxels but was not absorbed, check if it will arrive at the detector and tally its energy:
      tally_image(&energy, &position, &direction, &scatter_state, image, &source_data_SHARED, &detector_data_SHARED);
    }
  }   // [Continue with a new history]

}   // [All tracks simulated for this kernel call: return to CPU]






////////////////////////////////////////////////////////////////////////////////
//!  Tally the dose deposited in the voxels.
//!  This function is called whenever a particle suffers a Compton or photoelectric
//!  interaction. It is not necessary to call this function if the dose tally
//!  was disabled in the input file (ie, dose_ROI_x_max_CONST < 0).
//!  Electrons are not transported in MC-GPU and therefore we are approximating
//!  that the dose is equal to the KERMA (energy released by the photons alone).
//!  This approximation is acceptable when there is electronic equilibrium and when
//!  the range of the secondary electrons is shorter than the voxel size. Usually the
//!  doses will be acceptable for photon energies below 1 MeV. The dose estimates may
//!  not be accurate at the interface of low density volumes.
//!
//!  We need to use atomicAdd() in the GPU to prevent that multiple threads update the 
//!  same voxel at the same time, which would result in a lose of information.
//!  This is very improbable when using a large number of voxels but gives troubles 
//!  with a simple geometries with few voxels (in this case the atomicAdd will slow 
//!  down the code because threads will update the voxel dose secuentially).
//!
//!
//!       @param[in] Edep   Energy deposited in the interaction
//!       @param[in] voxel_coord   Voxel coordinates, needed to check if particle located inside the input region of interest (ROI)
//!       @param[out] voxels_Edep   ulonglong2 array containing the 3D voxel dose and dose^2 (ie, uncertainty) as unsigned integers scaled by SCALE_eV.
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline 
void tally_voxel_energy_deposition(float* Edep, short3* voxel_coord, ulonglong2* voxels_Edep)
{

    // !!DeBuG!! Maybe it would be faster to store a 6 element struct and save temp copy?? struct_short_int_x6_align16  dose_ROI_size = dose_ROI_size_CONST;   // Get ROI coordinates from GPU constant memory and store temporal copy
  
  if((voxel_coord->x < dose_ROI_x_min_CONST) || (voxel_coord->x > dose_ROI_x_max_CONST) ||
     (voxel_coord->y < dose_ROI_y_min_CONST) || (voxel_coord->y > dose_ROI_y_max_CONST) ||
     (voxel_coord->z < dose_ROI_z_min_CONST) || (voxel_coord->z > dose_ROI_z_max_CONST))
    {
      return;   // -- Particle outside the ROI: return without tallying anything.
    }

  // -- Particle inside the ROI: tally Edep.
  register int DX = 1 + (int)(dose_ROI_x_max_CONST - dose_ROI_x_min_CONST);
  register int num_voxel = (int)(voxel_coord->x-dose_ROI_x_min_CONST) + ((int)(voxel_coord->y-dose_ROI_y_min_CONST))*DX + ((int)(voxel_coord->z-dose_ROI_z_min_CONST))*DX*(1 + (int)(dose_ROI_y_max_CONST-dose_ROI_y_min_CONST));
  
   #ifdef USING_CUDA
     atomicAdd(&voxels_Edep[num_voxel].x, __float2ull_rn((*Edep)*SCALE_eV) );    // Energy deposited at the voxel, scaled by the factor SCALE_eV and rounded.
     atomicAdd(&voxels_Edep[num_voxel].y, __float2ull_rn((*Edep)*(*Edep)) );     // (not using SCALE_eV for std_dev to prevent overflow)           
   #else
     voxels_Edep[num_voxel].x += (unsigned long long int)((*Edep)*SCALE_eV + 0.5f);
     voxels_Edep[num_voxel].y += (unsigned long long int)((*Edep)*(*Edep) + 0.5f);
   #endif
          
  return;
}


////////////////////////////////////////////////////////////////////////////////
//!  Tally a radiographic projection image.
//!  This function is called whenever a particle escapes the voxelized volume.
//!  The code checks if the particle would arrive at the detector if it kept
//!  moving in a straight line after exiting the voxels (assuming vacuum enclosure).
//!  An ideal image formation model is implemented: each pixel counts the total energy
//!  of the x rays that enter the pixel (100% detection efficiency for any energy).
//!  The image due to primaries and different kinds of scatter is tallied separately.
//!
//!  In the GPU, and atomicAdd() function is used to make sure that multiple threads do
//!  not update the same pixel at the same time, which would result in a lose of information.
//!  Since the atomicAdd function is only available for 'unsigned long long int' data,
//!  the float pixel values are scaled by a factor "SCALE_eV" defined in the header file
//!  (eg, #define SCALE_eV 10000.0f) and stored as unsigned long long integers in main
//!  memory.
//!
//!  WARNING! If the total tallied signal (for all particles) is larger than "1.8e19/SCALE_eV",
//!    there will be a bit overflow and the value will be reset to 0 giving bogus results.
//!
//!  WARNING! The detector plane should be located outside the voxels bounding box. However, since
//!    the particles are moved outside the bbox in the last step, they could cross the detector 
//!    plane anyway. If the particles are less than 2.0 cm behind the detector, they are moved 
//!    back and detected. Therefore the detector can be a few cm inside the bbox and still work.
//!    If the Woodcock mean free path is larger than the distance from the bbox to the detector, 
//!    we may lose some particles behind the detector!
//!
//!
//!       @param[in] energy   X-ray energy
//!       @param[in] position   Particle position
//!       @param[in] direction   Particle direction (cosine vectors)
//!       @param[in] scatter_state  Flag marking primaries, single Compton, single Rayleigh or multiple scattered radiation
//!       @param[out] image   Integer array containing the image, ie, the pixel values (in tenths of meV)
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline void tally_image(float* energy, float3* position, float3* direction, signed char* scatter_state, unsigned long long int* image, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED)
{
  float dist_detector, rotated_position;

  if (detector_data_SHARED->rotation_flag == 1)    // -->  Initial source direction is not (0,1,0): detector has to be rotated to +Y to find the pixel number
  {
    
    // *** Skip particles not moving towards the detector. 
    //       (a) Skip particles that were deflected more than 90 deg from the original source direction (backscatter).
    //       (b) Skip particles located more than 10 cm behind the detector 
    //       (c) Skip particles for which the direction to the detector is way bigger than SDD (likely to intersect the plane outside the pixel region).
                  // !!DeBuG!! NOTE: This may give problems for big detectors very close to the source
                  
    //      !!DeBuG!! Particles located after the detector will be moved back to the surface of the detector, but 10 cm maximum!!
    //                In this way the detector can intersect the voxels bbox or be located right on the surface of the bbox: the particles will be 
    //                transported across the detector and until a little after the end of the bbox in the last step, but then moved back.
    //                This algorithm will give correct results ONLY when the detector intersects just slightly the air space around the phantom,
    //                so that the interactions after the detector are not significant (this happens sometimes using oblique beams).
    //                I could remove particles after the detector using "if (dist_detector<0.0f) return;".

    //  (a) Calculate the angle between the particle and the initial direction (dot product): reject particle if cos_angle < cos(89)==0 (angle>89deg):
    //      [Extra parenthesis are coded to suggest to the compiler the use of intrinsic multiply-add operations].

    register float cos_angle = direction->x * source_data_SHARED->direction.x +
                              (direction->y * source_data_SHARED->direction.y +
                              (direction->z * source_data_SHARED->direction.z));    
    if (cos_angle < 0.025f)
      return;  // Reject particle: Angle larger than 89 deg --> particle moving parallel to the detector or backwards towards the source!

    //   (b) Find the distance from the current particle location (likely just after the surface of the voxel bbox) to the intersection with the detector plane:
    dist_detector = ( source_data_SHARED->direction.x * (detector_data_SHARED->center.x - position->x) +
                     (source_data_SHARED->direction.y * (detector_data_SHARED->center.y - position->y) +
                     (source_data_SHARED->direction.z * (detector_data_SHARED->center.z - position->z))) ) / cos_angle;

                        
                     
// !!DeBuG!!  IF's below (used in v1.2) are not needed when checking the x ray angle:
//   if (dist_detector < -10.0f)   // !!DeBuG!! Is 10 cm enough or too much? Should I use 0? or allow any distance?
//      return;  // !!DeBuG!! Reject particles located more than 10 cm behind the detector. 10 cm was selected arbitrarily. Woodcock MFP for x-rays in bone: MFP 200 keV photons in bone ==> 4 cm.
//      
//    if (fabsf(dist_detector)>(2.1f*detector_data_CONST.sdd))          
//      return;  // Reject particle: distance to the detector plane too large, the particle is likely to travel almost parallel to the detector and will not be detected.

            
    // *** Translate the particle to the detector plane (we assume the detector is completely absorbent: 100% detection efficiency):
    position->x = position->x + dist_detector * direction->x;
    position->y = position->y + dist_detector * direction->y;
    position->z = position->z + dist_detector * direction->z;

    // *** Rotate the particle position vector to the default reference system where the detector is perpendicular to the +Y axis, then find out if the particle is located inside a pixel:
    #ifdef USING_CUDA
      rotated_position = detector_data_SHARED->rot_inv[0]*position->x + detector_data_SHARED->rot_inv[1]*position->y + detector_data_SHARED->rot_inv[2]*position->z;  // X coordinate
      int pixel_coord_x = __float2int_rd((rotated_position - detector_data_SHARED->corner_min_rotated_to_Y.x) * detector_data_SHARED->inv_pixel_size_X);    // Using CUDA intrinsic function to convert float to integer rounding down (towards minus infinite)
      if ((pixel_coord_x>-1)&&(pixel_coord_x<detector_data_SHARED->num_pixels.x))
      {
        rotated_position = detector_data_SHARED->rot_inv[6]*position->x + detector_data_SHARED->rot_inv[7]*position->y + detector_data_SHARED->rot_inv[8]*position->z;  // Z coordinate
        int pixel_coord_z = __float2int_rd((rotated_position - detector_data_SHARED->corner_min_rotated_to_Y.z) * detector_data_SHARED->inv_pixel_size_Z);
        if ((pixel_coord_z>-1)&&(pixel_coord_z<detector_data_SHARED->num_pixels.y))
        {
          // -- Particle enters the detector! Tally the particle energy in the corresponding pixel (in tenths of meV):
          //    Using a CUDA atomic function (not available for global floats yet) to read and increase the pixel value in a single instruction, blocking interferences from other threads.
          //    The offset for the primaries or scatter images are calculated considering that:
          //      scatter_state=0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter.
          atomicAdd(( image +                                                               // Pointer to beginning of image array
                    (int)(*scatter_state) * detector_data_SHARED->total_num_pixels +         // Offset to corresponding scatter image
                    (pixel_coord_x + pixel_coord_z*(detector_data_SHARED->num_pixels.x)) ),  // Offset to the corresponding pixel
                    __float2ull_rn((*energy)*SCALE_eV) );     // Energy arriving at the pixel, scaled by the factor SCALE_eV and rounded.
                                                              // The maximum unsigned long long int value is ~1.8e19:
        }
      }
    #else
      // CPU version (not using CUDA intrinsics: atomicAdd, fast type casting)
      rotated_position = detector_data_SHARED->rot_inv[0]*position->x + detector_data_SHARED->rot_inv[1]*position->y + detector_data_SHARED->rot_inv[2]*position->z;  // X coordinate
      
      float pixel_coord_x = floor((rotated_position - detector_data_SHARED->corner_min_rotated_to_Y.x)*detector_data_SHARED->inv_pixel_size_X);   // Using float+floor instead of INT to avoid truncation errors for positive and negative values
      if ( (pixel_coord_x>-0.1f) && (pixel_coord_x<(detector_data_SHARED->num_pixels.x-0.1f)) )    // Rejecting values negative or bigger than the image size
      {
        rotated_position = detector_data_SHARED->rot_inv[6]*position->x + detector_data_SHARED->rot_inv[7]*position->y + detector_data_SHARED->rot_inv[8]*position->z;  // Z coordinate
        float pixel_coord_z = floor((rotated_position - detector_data_SHARED->corner_min_rotated_to_Y.z)*detector_data_SHARED->inv_pixel_size_Z);
        if ( (pixel_coord_z>-0.1f) && (pixel_coord_z<(detector_data_SHARED->num_pixels.y-0.1f)) )
          image[(int)(((float)*scatter_state)*detector_data_SHARED->total_num_pixels + pixel_coord_x + pixel_coord_z*detector_data_SHARED->num_pixels.x  +  0.0001f)]
             += (unsigned long long int)((*energy)*SCALE_eV + 0.5f);   // Tally the particle energy in the pixel. This instruction is not thread-safe, but it is ok in sequential CPU code.          
      }
    #endif
  }
  else  // (detector_data_SHARED->rotation_flag != 1) -->  Initial source direction is (0,1,0): pixel number and distance can be found easily
  {  
    if (direction->y < 0.0001f)
      return;  // *** Reject particles not moving towards the detector plane at +Y.

    dist_detector = (detector_data_SHARED->center.y - position->y)/(direction->y);  // Distance to the intersection with the detector at +Y.
  
      // !!DeBuG!! IF below (v1.2) not needed when checking the angle
      //     if (dist_detector>(2.1f*detector_data_SHARED->sdd)) return;  
     
    
    #ifdef USING_CUDA
    int pixel_coord_x = __float2int_rd((position->x + dist_detector*direction->x - detector_data_SHARED->corner_min_rotated_to_Y.x)*detector_data_SHARED->inv_pixel_size_X);
    if ((pixel_coord_x>-1)&&(pixel_coord_x<detector_data_SHARED->num_pixels.x))
    {
      int pixel_coord_z = __float2int_rd((position->z + dist_detector*direction->z - detector_data_SHARED->corner_min_rotated_to_Y.z)*detector_data_SHARED->inv_pixel_size_Z);
      if ((pixel_coord_z>-1)&&(pixel_coord_z<detector_data_SHARED->num_pixels.y))
        atomicAdd( ( image +                                                                // Pointer to beginning of image array
                     (int)(*scatter_state) * detector_data_SHARED->total_num_pixels +         // Offset to corresponding scatter image
                     (pixel_coord_x + pixel_coord_z*(detector_data_SHARED->num_pixels.x)) ),  // Offset to the corresponding pixel
                   __float2ull_rn((*energy)*SCALE_eV) );    // Energy arriving at the pixel, scaled by the factor SCALE_eV and rounded.
    }
    #else

    // --Calculate the pixel the xray enters, truncating towards minus infinite and making sure the conversion to int is safe:
    float pixel_coord_x = floor((position->x + dist_detector*direction->x - detector_data_SHARED->corner_min_rotated_to_Y.x)*detector_data_SHARED->inv_pixel_size_X);

    if ( (pixel_coord_x>-0.1f) && (pixel_coord_x<(detector_data_SHARED->num_pixels.x-0.1f)) )
    {
      float pixel_coord_z = floor((position->z + dist_detector*direction->z - detector_data_SHARED->corner_min_rotated_to_Y.z)*detector_data_SHARED->inv_pixel_size_Z);
      if ( (pixel_coord_z>-0.1f) && (pixel_coord_z<(detector_data_SHARED->num_pixels.y-0.1f)) )
        image[(int)(((float)*scatter_state)*detector_data_SHARED->total_num_pixels + pixel_coord_x + pixel_coord_z*detector_data_SHARED->num_pixels.x  +  0.0001f)]
           += (unsigned long long int)((*energy)*SCALE_eV + 0.5f);    // Truncate the pixel number to INT and round the energy value
    }
    #endif
  }

}



////////////////////////////////////////////////////////////////////////////////
//!  Source that creates primary x rays, according to the defined source model.
//!  The particles are automatically moved to the surface of the voxel bounding box,
//!  to start the tracking inside a real material. If the sampled particle do not
//!  enter the voxels, it is init in the focal spot and the main program will check
//!  if it arrives at the detector or not.
//!
//!       @param[in] source_data   Structure describing the source.
//!       @param[in] source_energy_data_CONST   Global variable in constant memory space describing the source energy spectrum.
//!       @param[out] position   Initial particle position (particle transported inside the voxel bbox).
//!       @param[out] direction   Sampled particle direction (cosine vectors).
//!       @param[out] energy   Sampled energy of the new x ray.
//!       @param[in] seed   Current seed of the random number generator, requiered to sample the movement direction.
//!       @param[out] absvox   Set to <0 if primary particle will not cross the voxels, not changed otherwise (>0).
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline void source(float3* position, float3* direction, float* energy, int2* seed, int* absvox, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED)
{
  // *** Sample the initial x-ray energy following the input energy spectrum using the Walker aliasing algorithm from PENELOPE:
      // The following code is equivalent to calling the function "seeki_walker": int sampled_bin = seeki_walker(source_data_CONST.espc_cutoff, source_data_CONST.espc_alias, ranecu(seed), source_data_CONST.num_bins_espc);      
  int sampled_bin;
  float RN = ranecu(seed) * source_energy_data_CONST.num_bins_espc;    // Find initial interval (array starting at 0):   
  #ifdef USING_CUDA
    int int_part = __float2int_rd(RN);                          //   -- Integer part (round down)
  #else
    int int_part = (int)(RN);
  #endif
  float fraction_part = RN - ((float)int_part);                 //   -- Fractional part
  if (fraction_part < source_energy_data_CONST.espc_cutoff[int_part])  // Check if we are in the aliased part
    sampled_bin = int_part;                                     // Below the cutoff: return current value
  else
    sampled_bin = (int)source_energy_data_CONST.espc_alias[int_part];  // Above the cutoff: return alias
  
  // Linear interpolation of the final energy within the sampled energy bin:
  *energy = source_energy_data_CONST.espc[sampled_bin] + ranecu(seed) * (source_energy_data_CONST.espc[sampled_bin+1] - source_energy_data_CONST.espc[sampled_bin]);   
      
 
   // *** Sample the initial direction:
   
  do   //  Iterate sampling if the sampled direction is not acceptable to get a square field at the given phi (rejection sampling): force square field for any phi!!
  {
    //     Using the algorithm used in PENMAIN.f, from penelope 2008 (by F. Salvat).
    direction->z = source_data_SHARED->cos_theta_low + ranecu(seed)*source_data_SHARED->D_cos_theta;     // direction->z = w = cos(theta_sampled)
    register float phi_sampled = source_data_SHARED->phi_low + ranecu(seed)*source_data_SHARED->D_phi;
    register float sin_theta_sampled = sqrtf(1.0f - direction->z*direction->z);
    float sinphi_sampled, cosphi_sampled;
    
    #ifdef USING_CUDA
      sincos(phi_sampled, &sinphi_sampled,&cosphi_sampled);    // Calculate the SIN and COS at the same time.
    #else
      sinphi_sampled = sin(phi_sampled);   // Some CPU compilers will be able to use "sincos", but let's be safe.
      cosphi_sampled = cos(phi_sampled);
    #endif       
    
    direction->y = sin_theta_sampled * sinphi_sampled;
    direction->x = sin_theta_sampled * cosphi_sampled;
  }
  while( fabsf(direction->z/(direction->y+1.0e-7f)) > source_data_SHARED->max_height_at_y1cm );  // !!DeBuG!! Force square field for any phi by rejection sampling!! Is it necessary to use the "+1.0e-7f" to prevent possible division by zero???
    

  if (detector_data_SHARED->rotation_flag == 1)
  {
    // --Initial beam not pointing to (0,1,0), apply rotation:
    register float direction_x_tmp = direction->x;
    register float direction_y_tmp = direction->y;
    direction->x = source_data_SHARED->rot_fan[0]*direction_x_tmp + source_data_SHARED->rot_fan[1]*direction_y_tmp + source_data_SHARED->rot_fan[2]*direction->z;
    direction->y = source_data_SHARED->rot_fan[3]*direction_x_tmp + source_data_SHARED->rot_fan[4]*direction_y_tmp + source_data_SHARED->rot_fan[5]*direction->z;
    direction->z = source_data_SHARED->rot_fan[6]*direction_x_tmp + source_data_SHARED->rot_fan[7]*direction_y_tmp + source_data_SHARED->rot_fan[8]*direction->z;
  }

  // Initialize x ray position at focal spot before translation into bbox. Particle stays in focal spot if no interaction found:
  position->x = source_data_SHARED->position.x;
  position->y = source_data_SHARED->position.y;
  position->z = source_data_SHARED->position.z;
      
  move_to_bbox(position, direction, voxel_data_CONST.size_bbox, absvox);  // Move the particle inside the voxel bounding box.
}



////////////////////////////////////////////////////////////////////////////////
//!  Functions that moves a particle inside the voxelized geometry bounding box.
//!  An EPSILON distance is added to make sure the particles will be clearly inside the bbox, 
//!  not exactly on the surface. 
//!
//!  This algorithm makes the following assumtions:
//!     - The back lower vertex of the voxel bounding box is always located at the origin: (x0,y0,z0)=(0,0,0).
//!     - The initial value of "position" corresponds to the focal spot location.
//!     - When a ray is not pointing towards the bbox plane that it should cross according to the sign of the direction,
//!       I assign a distance to the intersection =0 instead of the real negative distance. The wall that will be 
//!       crossed to enter the bbox is always the furthest and therefore a 0 distance will never be used except
//!       in the case of a ray starting inside the bbox or outside the bbox and not pointing to any of the 3 planes. 
//!       In this situation the ray will be transported a 0 distance, meaning that it will stay at the focal spot.
//!
//!  (Interesting information on ray-box intersection: http://tog.acm.org/resources/GraphicsGems/gems/RayBox.c)
//!
//!       @param[in,out] position Particle position: initially set to the focal spot, returned transported inside the voxel bbox.
//!       @param[out] direction   Sampled particle direction (cosine vectors).
//!       @param[out] intersection_flag   Set to <0 if particle outside bbox and will not cross the voxels, not changed otherwise.
//!       @param[out] size_bbox   Size of the bounding box.
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline void move_to_bbox(float3* position, float3* direction, float3 size_bbox, int* intersection_flag)
{
  float dist_y, dist_x, dist_z;

  // -Distance to the nearest Y plane:
  if ((direction->y) > EPS_SOURCE)   // Moving to +Y: check distance to y=0 plane
  {
    // Check Y=0 (bbox wall):
    if (position->y > 0.0f)  // The input position must correspond to the focal spot => position->y == source_data_CONST.position[*num_p].y
      dist_y = 0.0f;  // No intersection with this plane: particle inside or past the box  
          // The actual distance would be negative but we set it to 0 bc we will not move the particle if no intersection exist.
    else
      dist_y = EPS_SOURCE + (-position->y)/(direction->y);    // dist_y > 0 for sure in this case
  }
  else if ((direction->y) < NEG_EPS_SOURCE)
  {
    // Check Y=voxel_data_CONST.size_bbox.y:
    if (position->y < size_bbox.y)
      dist_y = 0.0f;  // No intersection with this plane
    else
      dist_y = EPS_SOURCE + (size_bbox.y - position->y)/(direction->y);    // dist_y > 0 for sure in this case
  }
  else   // (direction->y)~0
    dist_y = NEG_INF;   // Particle moving parallel to the plane: no interaction possible (set impossible negative dist = -INFINITE)

  // -Distance to the nearest X plane:
  if ((direction->x) > EPS_SOURCE)
  {
    // Check X=0:
    if (position->x > 0.0f)
      dist_x = 0.0f;
    else  
      dist_x = EPS_SOURCE + (-position->x)/(direction->x);    // dist_x > 0 for sure in this case
  }
  else if ((direction->x) < NEG_EPS_SOURCE)
  {
    // Check X=voxel_data_CONST.size_bbox.x:
    if (position->x < size_bbox.x)
      dist_x = 0.0f;
    else  
      dist_x = EPS_SOURCE + (size_bbox.x - position->x)/(direction->x);    // dist_x > 0 for sure in this case
  }
  else
    dist_x = NEG_INF;

  // -Distance to the nearest Z plane:
  if ((direction->z) > EPS_SOURCE)
  {
    // Check Z=0:
    if (position->z > 0.0f)
      dist_z = 0.0f;
    else
      dist_z = EPS_SOURCE + (-position->z)/(direction->z);    // dist_z > 0 for sure in this case
  }
  else if ((direction->z) < NEG_EPS_SOURCE)
  {
    // Check Z=voxel_data_CONST.size_bbox.z:
    if (position->z < size_bbox.z)
      dist_z = 0.0f;
    else
      dist_z = EPS_SOURCE + (size_bbox.z - position->z)/(direction->z);    // dist_z > 0 for sure in this case
  }
  else
    dist_z = NEG_INF;

  
  // -- Find the longest distance plane, which is the one that has to be crossed to enter the bbox.
  //    Storing the maximum distance in variable "dist_z". Distance will be =0 if no intersection exists or 
  //    if the x ray is already inside the bbox.
  if ( (dist_y>dist_x) && (dist_y>dist_z) )
    dist_z = dist_y;      // dist_z == dist_max 
  else if (dist_x>dist_z)
    dist_z = dist_x;
// else
//   dist_max = dist_z;
    
  // -- Move particle from the focal spot (current location) to the bbox wall surface (slightly inside):
  position->x += dist_z * direction->x;
  position->y += dist_z * direction->y;
  position->z += dist_z * direction->z;      
  
  // Check if the new position is outside the bbox. If true, the particle must be moved back to the focal spot location:
  if ( (position->x < 0.0f) || (position->x > size_bbox.x) || 
       (position->y < 0.0f) || (position->y > size_bbox.y) || 
       (position->z < 0.0f) || (position->z > size_bbox.z) )
  {
    position->x -= dist_z * direction->x;  // Reject new position undoing the previous translation
    position->y -= dist_z * direction->y;
    position->z -= dist_z * direction->z;
    (*intersection_flag) = -111;  // Particle outside the bbox AND not pointing to the bbox: set absvox<0 to skip interaction sampling.
  }
}


////////////////////////////////////////////////////////////////////////////////


//!  Upper limit of the number of random values sampled in a single track.
#define  LEAP_DISTANCE     256
//!  Multipliers and moduli for the two MLCG in RANECU.
#define  a1_RANECU       40014
#define  m1_RANECU  2147483563
#define  a2_RANECU       40692
#define  m2_RANECU  2147483399
////////////////////////////////////////////////////////////////////////////////
//! Initialize the pseudo-random number generator (PRNG) RANECU to a position
//! far away from the previous history (leap frog technique).
//!
//! Each calculated seed initiates a consecutive and disjoint sequence of
//! pseudo-random numbers with length LEAP_DISTANCE, that can be used to
//! in a parallel simulation (Sequence Splitting parallelization method).
//! The basic equation behind the algorithm is:
//!    S(i+j) = (a**j * S(i)) MOD m = [(a**j MOD m)*S(i)] MOD m  ,
//! which is described in:
//!   P L'Ecuyer, Commun. ACM 31 (1988) p.742
//!
//! This function has been adapted from "seedsMLCG.f", see:
//!   A Badal and J Sempau, Computer Physics Communications 175 (2006) p. 440-450
//!
//!       @param[in] history   Particle bach number.
//!       @param[in] seed_input   Initial PRNG seed input (used to initiate both MLCGs in RANECU).
//!       @param[out] seed   Initial PRNG seeds for the present history.
//!
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline void init_PRNG(int history_batch, int histories_per_thread, int seed_input, int2* seed)
{
  // -- Move the RANECU generator to a unique position for the current batch of histories:
  //    I have to use an "unsigned long long int" value to represent all the simulated histories in all previous batches
  //    The maximum unsigned long long int value is ~1.8e19: if history >1.8e16 and LEAP_DISTANCE==1000, 'leap' will overflow.
  // **** 1st MLCG:
  unsigned long long int leap = ((unsigned long long int)(history_batch+1))*(histories_per_thread*LEAP_DISTANCE);
  int y = 1;
  int z = a1_RANECU;
  // -- Calculate the modulo power '(a^leap)MOD(m)' using a divide-and-conquer algorithm adapted to modulo arithmetic
  for(;;)
  {
    // (A2) Halve n, and store the integer part and the residue
    if (0!=(leap&01))  // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
    {
      leap >>= 1;     // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);  
      y = abMODm(m1_RANECU,z,y);      // (A3) Multiply y by z:  y = [z*y] MOD m
      if (0==leap) break;         // (A4) leap==0? ==> finish
    }
    else           // (leap is even)
    {
      leap>>= 1;     // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
    }
    z = abMODm(m1_RANECU,z,z);        // (A5) Square z:  z = [z*z] MOD m
  }
  // AjMODm1 = y;                 // Exponentiation finished:  AjMODm = expMOD = y = a^j

  // -- Compute and display the seeds S(i+j), from the present seed S(i), using the previously calculated value of (a^j)MOD(m):
  //         S(i+j) = [(a**j MOD m)*S(i)] MOD m
  //         S_i = abMODm(m,S_i,AjMODm)
  seed->x = abMODm(m1_RANECU, seed_input, y);     // Using the input seed as the starting seed

  // **** 2nd MLCG (repeating the previous calculation for the 2nd MLCG parameters):
  leap = ((unsigned long long int)(history_batch+1))*(histories_per_thread*LEAP_DISTANCE);
  y = 1;
  z = a2_RANECU;
  for(;;)
  {
    // (A2) Halve n, and store the integer part and the residue
    if (0!=(leap&01))  // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
    {
      leap >>= 1;     // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);
      y = abMODm(m2_RANECU,z,y);      // (A3) Multiply y by z:  y = [z*y] MOD m
      if (0==leap) break;         // (A4) leap==0? ==> finish
    }
    else           // (leap is even)
    {
      leap>>= 1;     // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
    }
    z = abMODm(m2_RANECU,z,z);        // (A5) Square z:  z = [z*z] MOD m
  }
  // AjMODm2 = y;
  seed->y = abMODm(m2_RANECU, seed_input, y);     // Using the input seed as the starting seed
}



/////////////////////////////////////////////////////////////////////
//!  Calculate "(a1*a2) MOD m" with 32-bit integers and avoiding
//!  the possible overflow, using the Russian Peasant approach
//!  modulo m and the approximate factoring method, as described
//!  in:  L'Ecuyer and Cote, ACM Trans. Math. Soft. 17 (1991).
//!
//!  This function has been adapted from "seedsMLCG.f", see: 
//!  Badal and Sempau, Computer Physics Communications 175 (2006)
//!
//!       @param[in] m,a,s  MLCG parameters
//!       @return   (a1*a2) MOD m   
//
//    Input:          0 < a1 < m                                  
//                    0 < a2 < m                                  
//
//    Return value:  (a1*a2) MOD m                                
//
/////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__ __host__    // Function will be callable from host and also from device
#endif
inline int abMODm(int m, int a, int s)
{
  // CAUTION: the input parameters are modified in the function but should not be returned to the calling function! (pass by value!)
  int q, k;
  int p = -m;            // p is always negative to avoid overflow when adding

  // ** Apply the Russian peasant method until "a =< 32768":
  while (a>32768)        // We assume '32' bit integers (4 bytes): 2^(('32'-2)/2) = 32768
  {
    if (0!=(a&1))        // Store 's' when 'a' is odd     Equivalent code:   if (1==(a%2))
    {
      p += s;
      if (p>0) p -= m;
    }
    a >>= 1;             // Half a (move bits 1 position right)   Equivalent code: a = a/2;
    s = (s-m) + s;       // Double s (MOD m)
    if (s<0) s += m;     // (s is always positive)
  }

  // ** Employ the approximate factoring method (a is small enough to avoid overflow):
  q = (int) m / a;
  k = (int) s / q;
  s = a*(s-k*q)-k*(m-q*a);
  while (s<0)
    s += m;

  // ** Compute the final result:
  p += s;
  if (p<0) p += m;

  return p;
}



////////////////////////////////////////////////////////////////////////////////
//! Pseudo-random number generator (PRNG) RANECU returning a float value
//! (single precision version).
//!
//!       @param[in,out] seed   PRNG seed (seed kept in the calling function and updated here).
//!       @return   PRN double value in the open interval (0,1)
//!
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__ 
#endif
inline float ranecu(int2* seed)
{
  int i1 = (int)(seed->x/53668);
  seed->x = 40014*(seed->x-i1*53668)-i1*12211;

  int i2 = (int)(seed->y/52774);
  seed->y = 40692*(seed->y-i2*52774)-i2*3791;

  if (seed->x < 0) seed->x += 2147483563;
  if (seed->y < 0) seed->y += 2147483399;

  i2 = seed->x-seed->y;
  if (i2 < 1) i2 += 2147483562;


#ifdef USING_CUDA
  return (__int2float_rn(i2)*4.65661305739e-10f);        // 4.65661305739e-10 == 1/2147483563
#else
  return ((float)(i2)*4.65661305739e-10f);          
#endif

}


////////////////////////////////////////////////////////////////////////////////
//! Pseudo-random number generator (PRNG) RANECU returning a double value.
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__ 
#endif
inline double ranecu_double(int2* seed)
{
  int i1 = (int)(seed->x/53668);
  seed->x = 40014*(seed->x-i1*53668)-i1*12211;

  int i2 = (int)(seed->y/52774);
  seed->y = 40692*(seed->y-i2*52774)-i2*3791;

  if (seed->x < 0) seed->x += 2147483563;
  if (seed->y < 0) seed->y += 2147483399;

  i2 = seed->x-seed->y;
  if (i2 < 1) i2 += 2147483562;

#ifdef USING_CUDA
  return (__int2double_rn(i2)*4.6566130573917692e-10);
#else
  return ((double)(i2)*4.6566130573917692e-10);
#endif

}



////////////////////////////////////////////////////////////////////////////////
//! Find the voxel that contains the current position.
//! Report the voxel absolute index and the x,y,z indices.
//! The structure containing the voxel number and size is read from CONSTANT memory.
//!
//!       @param[in] position   Particle position
//!       @param[out] voxel_coord   Pointer to three integer values (short3*) that will store the x,y and z voxel indices.
//!       @return   Returns "absvox", the voxel number where the particle is
//!                 located (negative if position outside the voxel bbox).
//!
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline int locate_voxel(float3* position, short3* voxel_coord)
{

  if ( (position->y < EPS_SOURCE) || (position->y > (voxel_data_CONST.size_bbox.y - EPS_SOURCE)) ||
       (position->x < EPS_SOURCE) || (position->x > (voxel_data_CONST.size_bbox.x - EPS_SOURCE)) ||
       (position->z < EPS_SOURCE) || (position->z > (voxel_data_CONST.size_bbox.z - EPS_SOURCE)) )
  {
    // -- Particle escaped the voxelized geometry (using EPS_SOURCE to avoid numerical precision errors):      
     return -1;
  }
 
  // -- Particle inside the voxelized geometry, find current voxel:
  //    The truncation from float to integer could give troubles for negative coordinates but this will never happen thanks to the IF at the begining of this function.
  //    (no need to use the CUDA function to convert float to integer rounding down (towards minus infinite): __float2int_rd)
  
  register int voxel_coord_x, voxel_coord_y, voxel_coord_z;
#ifdef USING_CUDA
  voxel_coord_x = __float2int_rd(position->x * voxel_data_CONST.inv_voxel_size.x);  
  voxel_coord_y = __float2int_rd(position->y * voxel_data_CONST.inv_voxel_size.y);
  voxel_coord_z = __float2int_rd(position->z * voxel_data_CONST.inv_voxel_size.z);
#else
  voxel_coord_x = (int)(position->x * voxel_data_CONST.inv_voxel_size.x);     
  voxel_coord_y = (int)(position->y * voxel_data_CONST.inv_voxel_size.y);
  voxel_coord_z = (int)(position->z * voxel_data_CONST.inv_voxel_size.z);
#endif

  // Output the voxel coordinates as short int (2 bytes) instead of int (4 bytes) to save registers; avoid type castings in the calculation of the return value.
  voxel_coord->x = (short int) voxel_coord_x;
  voxel_coord->y = (short int) voxel_coord_y;
  voxel_coord->z = (short int) voxel_coord_z;
  
  return (voxel_coord_x + voxel_coord_y*(voxel_data_CONST.num_voxels.x) + voxel_coord_z*(voxel_data_CONST.num_voxels.x)*(voxel_data_CONST.num_voxels.y));  
}



//////////////////////////////////////////////////////////////////////
//!   Rotates a vector; the rotation is specified by giving
//!   the polar and azimuthal angles in the "self-frame", as
//!   determined by the vector to be rotated.
//!   This function is a literal translation from Fortran to C of
//!   PENELOPE (v. 2006) subroutine "DIRECT".
//!
//!    @param[in,out]  (u,v,w)  input vector (=d) in the lab. frame; returns the rotated vector components in the lab. frame
//!    @param[in]  costh  cos(theta), angle between d before and after turn
//!    @param[in]  phi  azimuthal angle (rad) turned by d in its self-frame
//
//    Output:
//      (u,v,w) -> rotated vector components in the lab. frame
//
//    Comments:
//      -> (u,v,w) should have norm=1 on input; if not, it is
//         renormalized on output, provided norm>0.
//      -> The algorithm is based on considering the turned vector
//         d' expressed in the self-frame S',
//           d' = (sin(th)cos(ph), sin(th)sin(ph), cos(th))
//         and then apply a change of frame from S' to the lab
//         frame. S' is defined as having its z' axis coincident
//         with d, its y' axis perpendicular to z and z' and its
//         x' axis equal to y'*z'. The matrix of the change is then
//                   / uv/rho    -v/rho    u \
//          S ->lab: | vw/rho     u/rho    v |  , rho=(u^2+v^2)^0.5
//                   \ -rho       0        w /
//      -> When rho=0 (w=1 or -1) z and z' are parallel and the y'
//         axis cannot be defined in this way. Instead y' is set to
//         y and therefore either x'=x (if w=1) or x'=-x (w=-1)
//////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline void rotate_double(float3* direction, double costh, double phi)   // !!DeBuG!! The direction vector is single precision but the rotation is performed in doule precision for increased accuracy.
{
  double DXY, NORM, cosphi, sinphi, SDT;
  DXY = direction->x*direction->x + direction->y*direction->y;
  
#ifdef USING_CUDA
  sincos(phi, &sinphi,&cosphi);   // Calculate the SIN and COS at the same time.
#else
  sinphi = sin(phi);   // Some CPU compilers will be able to use "sincos", but let's be safe.
  cosphi = cos(phi);
#endif   

  // ****  Ensure normalisation
  NORM = DXY + direction->z*direction->z;     // !!DeBuG!! Check if it is really necessary to renormalize in a real simulation!!
  if (fabs(NORM-1.0)>1.0e-14)
  {
    NORM = 1.0/sqrt(NORM);
    direction->x = NORM*direction->x;
    direction->y = NORM*direction->y;
    direction->z = NORM*direction->z;
    DXY = direction->x*direction->x + direction->y*direction->y;
  }
  if (DXY>1.0e-28)
  {
    SDT = sqrt((1.0-costh*costh)/DXY);
    float direction_x_in = direction->x;
    direction->x = direction->x*costh + SDT*(direction_x_in*direction->z*cosphi-direction->y*sinphi);
    direction->y = direction->y*costh+SDT*(direction->y*direction->z*cosphi+direction_x_in*sinphi);
    direction->z = direction->z*costh-DXY*SDT*cosphi;
  }
  else
  {
    SDT = sqrt(1.0-costh*costh);
    direction->y = SDT*sinphi;
    if (direction->z>0.0)
    {
      direction->x = SDT*cosphi;
      direction->z = costh;
    }
    else
    {
      direction->x =-SDT*cosphi;
      direction->z =-costh;
    }
  }
}


//////////////////////////////////////////////////////////////////////


//  ***********************************************************************
//  *   Translation of PENELOPE's "SUBROUTINE GRAa" from FORTRAN77 to C   *
//  ***********************************************************************
//!  Sample a Rayleigh interaction using the sampling algorithm
//!  used in PENELOPE 2006.
//!
//!       @param[in] energy   Particle energy (not modified with Rayleigh)
//!       @param[out] costh_Rayleigh   Cosine of the angular deflection
//!       @param[in] material  Current voxel material
//
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//  C  PENELOPE/PENGEOM (version 2006)                                     C
//  C    Copyright (c) 2001-2006                                           C
//  C    Universitat de Barcelona                                          C
//  C  Permission to use, copy, modify, distribute and sell this software  C
//  C  and its documentation for any purpose is hereby granted without     C
//  C  fee, provided that the above copyright notice appears in all        C
//  C  copies and that both that copyright notice and this permission      C
//  C  notice appear in all supporting documentation. The Universitat de   C
//  C  Barcelona makes no representations about the suitability of this    C
//  C  software for any purpose. It is provided "as is" without express    C
//  C  or implied warranty.                                                C
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline void GRAa(float *energy, double *costh_Rayleigh, int *mat, float *pmax_current, int2 *seed, struct rayleigh_struct* cgra)
{
/*  ****  Energy grid and interpolation constants for the current energy. */
    double  xmax = ((double)*energy) * 8.065535669099010e-5;       // 8.065535669099010e-5 == 2.0*20.6074/510998.918
    double x2max = min_value( (xmax*xmax) , ((double)cgra->xco[(*mat+1)*NP_RAYLEIGH - 1]) );   // Get the last tabulated value of xco for this mat
    
    if (xmax < 0.01)
    {
       do
       {
          *costh_Rayleigh = 1.0 - ranecu_double(seed) * 2.0;
       }
       while ( ranecu_double(seed) > (((*costh_Rayleigh)*(*costh_Rayleigh)+1.0)*0.5) );
       return;
    }

    for(;;)    // (Loop will iterate everytime the sampled value is rejected or above maximum)
    {
      double ru = ranecu_double(seed) * (double)(*pmax_current);    // Pmax for the current energy is entered as a parameter
 
/*  ****  Selection of the interval  (binary search within pre-calculated limits). */
      int itn = (int)(ru * (NP_RAYLEIGH-1));     // 'itn' will never reach the last interval 'NP_RAYLEIGH-1', but this is how RITA is implemented in PENELOPE
      int i__ = (int)cgra->itlco[itn + (*mat)*NP_RAYLEIGH];
      int j   = (int)cgra->ituco[itn + (*mat)*NP_RAYLEIGH];
      
      if ((j - i__) > 1)
      {
        do
        {
          register int k = (i__ + j)>>1;     // >>1 == /2 
          if (ru > cgra->pco[k -1 + (*mat)*NP_RAYLEIGH])
            i__ = k;
          else
            j = k;
        }
        while ((j - i__) > 1);
      }
       
/*  ****  Sampling from the rational inverse cumulative distribution. */
      int index = i__ - 1 + (*mat)*NP_RAYLEIGH;

      double rr = ru - cgra->pco[index];
      double xx;
      if (rr > 1e-16)
      {      
        double d__ = (double)(cgra->pco[index+1] - cgra->pco[index]);
        float aco_index = cgra->aco[index], bco_index = cgra->bco[index], xco_index = cgra->xco[index];   // Avoid multiple accesses to the same global variable

        xx = (double)xco_index + (double)(aco_index + 1.0f + bco_index)* d__* rr / (d__*d__ + (aco_index*d__ + bco_index*rr) * rr) * (double)(cgra->xco[index+1] - xco_index);
        
      }
      else
      {
        xx = cgra->xco[index];
      }
      
      if (xx < x2max)
      {
        // Sampled value below maximum possible value:
        *costh_Rayleigh = 1.0 - 2.0 * xx / x2max;   // !!DeBuG!! costh_Rayleigh in double precision, but not all intermediate steps are!?
        /*  ****  Rejection: */    
        if (ranecu_double(seed) < (((*costh_Rayleigh)*(*costh_Rayleigh) + 1.0)*0.5))
          break;   // Sample value not rejected! break loop and return.
      }
    }
} /* graa */



//////////////////////////////////////////////////////////////////////////


//  ***********************************************************************
//  *   Translation of PENELOPE's "SUBROUTINE GCOa"  from FORTRAN77 to C  *
//  ********************************************************************* *
//!  Random sampling of incoherent (Compton) scattering of photons, using 
//!  the sampling algorithm from PENELOPE 2006:
//!    Relativistic impulse approximation with analytical one-electron Compton profiles

// !!DeBuG!!  In penelope, Doppler broadening is not used for E greater than 5 MeV.
//            We don't use it in GPU to reduce the lines of code and prevent using COMMON/compos/ZT(M)

//!       @param[in,out] energy   incident and final photon energy (eV)
//!       @param[out] costh_Compton   cosine of the polar scattering angle
//!       @param[in] material   Current voxel material
//!       @param[in] seed   RANECU PRNG seed
//
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//  C  PENELOPE/PENGEOM (version 2006)                                     C
//  C    Copyright (c) 2001-2006                                           C
//  C    Universitat de Barcelona                                          C
//  C  Permission to use, copy, modify, distribute and sell this software  C
//  C  and its documentation for any purpose is hereby granted without     C
//  C  fee, provided that the above copyright notice appears in all        C
//  C  copies and that both that copyright notice and this permission      C
//  C  notice appear in all supporting documentation. The Universitat de   C
//  C  Barcelona makes no representations about the suitability of this    C
//  C  software for any purpose. It is provided "as is" without express    C
//  C  or implied warranty.                                                C
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//
//  ************************************************************************

#ifdef USING_CUDA
__device__
#endif
inline void GCOa(float *energy, double *costh_Compton, int *mat, int2 *seed, struct compton_struct* cgco_SHARED)
{
    float s, a1, s0, af, ek, ek2, ek3, tau, pzomc, taumin;
    float rn[MAX_SHELLS];
    double cdt1;

     // Some variables used in PENELOPE have been eliminated to save register: float aux, taum2, fpzmax, a, a2, ek1 ,rni, xqc, fpz, pac[MAX_SHELLS];

    int i__;
    int my_noscco = cgco_SHARED->noscco[*mat];    // Store the number of oscillators for the input material in a local variable
    
#ifndef USING_CUDA
    static int warning_flag_1 = -1, warning_flag_2 = -1, warning_flag_3 = -1;    // Write warnings for the CPU code, but only once.  !!DeBuG!!
#endif

    ek = *energy * 1.956951306108245e-6f;    // (1.956951306108245e-6 == 1.0/510998.918)
    ek2 = ek * 2.f + 1.f;
    ek3 = ek * ek;
    // ek1 = ek3 - ek2 - 1.;
    taumin = 1.f / ek2;
    // taum2 = taumin * taumin;
    a1 = logf(ek2);
    // a2 = a1 + ek * 2. * (ek + 1.) * taum2;    // a2 was used only once, code moved below


/*  ****  Incoherent scattering function for theta=PI. */

    s0 = 0.0f;
    for (i__ = 0; i__ < my_noscco; i__++)
    {
       register float temp = cgco_SHARED->uico[*mat + i__*MAX_MATERIALS];
       if (temp < *energy)
       {
         register float aux = *energy * (*energy - temp) * 2.f;
         #ifdef USING_CUDA
           pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) * rsqrtf(aux + aux + temp * temp) * 1.956951306108245e-6f;
             // 1.956951306108245e-6 = 1.0/510998.918f   // Version using the reciprocal of sqrt in CUDA: faster and more accurate!!
         #else
           pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) / (sqrtf(aux + aux + temp * temp) * 510998.918f);
         #endif
         if (pzomc > 0.0f)
           temp = (0.707106781186545f+pzomc*1.4142135623731f) * (0.707106781186545f+pzomc*1.4142135623731f);
         else
           temp = (0.707106781186545f-pzomc*1.4142135623731f) * (0.707106781186545f-pzomc*1.4142135623731f);

         temp = 0.5f * expf(0.5f - temp);    // Calculate EXP outside the IF to avoid branching

         if (pzomc > 0.0f)
            temp = 1.0f - temp;
                                
         s0 += cgco_SHARED->fco[*mat + i__*MAX_MATERIALS] * temp;
       }
    }
            
/*  ****  Sampling tau. */
    do
    {
      if (ranecu(seed)*/*a2=*/(a1+2.*ek*(ek+1.f)*taumin*taumin) < a1)
      { 
        tau = powf(taumin, ranecu(seed));    // !!DeBuG!!  "powf()" has a big error (7 ULP), the double version has only 2!! 
      }
      else
      {
        tau = sqrtf(1.f + ranecu(seed) * (taumin * taumin - 1.f));
      }

      cdt1 = (double)(1.f-tau) / (((double)tau)*((double)*energy)*1.956951306108245e-6);    // !!DeBuG!! The sampled COS will be double precision, but TAU is not!!!

      if (cdt1 > 2.0) cdt1 = 1.99999999;   // !!DeBuG!! Make sure that precision error in POW, SQRT never gives cdt1>2 ==> costh_Compton<-1
      
  /*  ****  Incoherent scattering function. */
      s = 0.0f;
      for (i__ = 0; i__ < my_noscco; i__++)
      {
        register float temp = cgco_SHARED->uico[*mat + i__*MAX_MATERIALS];
        if (temp < *energy)
        {
          register float aux = (*energy) * (*energy - temp) * ((float)cdt1);

          if ((aux>1.0e-12f)||(temp>1.0e-12f))  // !!DeBuG!! Make sure the SQRT argument is never <0, and that we never get 0/0 -> NaN when aux=temp=0 !!
          {
         #ifdef USING_CUDA
           pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) * rsqrtf(aux + aux + temp * temp) * 1.956951306108245e-6f;
             // 1.956951306108245e-6 = 1.0/510998.918f   //  Version using the reciprocal of sqrt in CUDA: faster and more accurate!!
         #else
           pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) / (sqrtf(aux + aux + temp * temp) * 510998.918f);
         #endif

          }
          else
          {
            pzomc = 0.002f;    // !!DeBuG!! Using a rough approximation to a sample value of pzomc found using pure double precision: NOT RIGUROUS! But this code is expected to be used very seldom, only in extreme cases.
            #ifndef USING_CUDA
            if (warning_flag_1<0)
            {
               warning_flag_1 = +1;  // Disable warning, do not show again
               printf("          [... Small numerical precision error detected computing \"pzomc\" in GCOa (this warning will not be repeated).]\n               i__=%d, aux=%.14f, temp=%.14f, pzomc(forced)=%.14f, uico=%.14f, energy=%.7f, cgco_SHARED->fj0=%.14f, mat=%d, cdt1=%.14lf\n", (int)i__, aux, temp, pzomc, cgco_SHARED->uico[*mat+i__*MAX_MATERIALS], *energy, cgco_SHARED->fj0[*mat+i__*MAX_MATERIALS], (int)*mat, cdt1);   // !!DeBuG!!
            }
            #endif                    
          }
          
          temp = pzomc * 1.4142135623731f;
          if (pzomc > 0.0f)
            temp = 0.5f - (temp + 0.70710678118654502f) * (temp + 0.70710678118654502f);   // Calculate exponential argument
          else
            temp = 0.5f - (0.70710678118654502f - temp) * (0.70710678118654502f - temp);

          temp = 0.5f * expf(temp);      // All threads will calculate the expf together
          
          if (pzomc > 0.0f)
            temp = 1.0f - temp;

          s += cgco_SHARED->fco[*mat + i__*MAX_MATERIALS] * temp;
          rn[i__] = temp;
        }        
      }
    } while( (ranecu(seed)*s0) > (s*(1.0f+tau*(/*ek1=*/(ek3 - ek2 - 1.0f)+tau*(ek2+tau*ek3)))/(ek3*tau*(tau*tau+1.0f))) );  //  ****  Rejection function

    *costh_Compton = 1.0 - cdt1;
        
/*  ****  Target electron shell. */
    for (;;)
    {
      register float temp = s*ranecu(seed);
      float pac = 0.0f;

      int ishell = my_noscco - 1;     // First shell will have number 0
      for (i__ = 0; i__ < (my_noscco-1); i__++)    // !!DeBuG!! Iterate to (my_noscco-1) only: the last oscillator is excited in case all other fail (no point in double checking) ??
      {
        pac += cgco_SHARED->fco[*mat + i__*MAX_MATERIALS] * rn[i__];   // !!DeBuG!! pac[] is calculated on the fly to save registers!
        if (pac > temp)       //  pac[] is calculated on the fly to save registers!  
        {
            ishell = i__;
            break;
        }
      }

    /*  ****  Projected momentum of the target electron. */
      temp = ranecu(seed) * rn[ishell];

      if (temp < 0.5f)
      {
        pzomc = (0.70710678118654502f - sqrtf(0.5f - logf(temp + temp))) / (cgco_SHARED->fj0[*mat + ishell * MAX_MATERIALS] * 1.4142135623731f);
      }
      else
      {
        pzomc = (sqrtf(0.5f - logf(2.0f - 2.0f*temp)) - 0.70710678118654502f) / (cgco_SHARED->fj0[*mat + ishell * MAX_MATERIALS] * 1.4142135623731f);
      }
      if (pzomc < -1.0f)
      {
        continue;      // re-start the loop
      }

  /*  ****  F(EP) rejection. */
      temp = tau * (tau - (*costh_Compton) * 2.f) + 1.f;       // this variable was originally called "xqc"
      
        // af = sqrt( max_value(temp,1.0e-30f) ) * (tau * (tau - *costh_Compton) / max_value(temp,1.0e-30f) + 1.f);  //!!DeBuG!! Make sure the SQRT argument is never <0, and that I don't divide by zero!!

      if (temp>1.0e-20f)   // !!DeBuG!! Make sure the SQRT argument is never <0, and that I don't divide by zero!!
      {
        af = sqrtf(temp) * (tau * (tau - ((float)(*costh_Compton))) / temp + 1.f);
      }
      else
      {
        // When using single precision, it is possible (but very uncommon) to get costh_Compton==1 and tau==1; then temp is 0 and 'af' can not be calculated (0/0 -> nan). Analysing the results obtained using double precision, we found that 'af' would be almost 0 in this situation, with an "average" about ~0.002 (this is just a rough estimation, but using af=0 the value would never be rejected below).

        af = 0.00200f;    // !!DeBuG!!
                
        #ifndef USING_CUDA
        if (warning_flag_2<0)
        {
            warning_flag_2 = +1;  // Disable warning, do not show again
            printf("          [... Small numerical precision error detected computing \"af\" in GCOa (this warning will not be repeated)].\n               xqc=%.14f, af(forced)=%.14f, tau=%.14f, costh_Compton=%.14lf\n", temp, af, tau, *costh_Compton);    // !!DeBuG!!
        }
        #endif
      }

      if (af > 0.0f)
      {
        temp = af * 0.2f + 1.f;    // this variable was originally called "fpzmax"
      }
      else
      {
        temp = 1.f - af * 0.2f;
      }
      
      if ( ranecu(seed)*temp < /*fpz =*/(af * max_value( min_value(pzomc,0.2f) , -0.2f ) + 1.f) )
      {
        break;
      }

    }

/*  ****  Energy of the scattered photon. */
    {
      register float t, b1, b2, temp;
      t = pzomc * pzomc;
      b1 = 1.f - t * tau * tau;
      b2 = 1.f - t * tau * ((float)(*costh_Compton));

      temp = sqrtf( fabsf(b2 * b2 - b1 * (1.0f - t)) );
      
          
      if (pzomc < 0.0f)
         temp *= -1.0f;

      // !Error! energy may increase (slightly) due to inacurate calculation!  !!DeBuG!!
      t = (tau / b1) * (b2 + temp);
      if (t > 1.0f)
      {
        #ifndef USING_CUDA

        #endif      
        #ifndef USING_CUDA
        if (warning_flag_3<0)
        {
            warning_flag_3 = +1;  // Disable warning, do not show again
            printf("\n          [... a Compton event tried to increase the x ray energy due to precision error. Keeping initial energy. (This warning will not be repeated.)]\n               scaling=%.14f, costh_Compton=%.14lf\n", t, *costh_Compton);   // !!DeBuG!!
        }
        #endif
        
        t = 1.0f; // !!DeBuG!! Avoid increasing energy by hand!!! not nice!!
      }

      (*energy) *= t;
       // (*energy) *= (tau / b1) * (b2 + temp);    //  Original PENELOPE code
    }
    
}  // [End subroutine GCOa]



////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//!  Tally the depose deposited inside each material.
//!  This function is called whenever a particle suffers a Compton or photoelectric
//!  interaction. The energy released in each interaction is added and later in the 
//!  report function the total deposited energy is divided by the total mass of the 
//!  material in the voxelized object to get the dose. This naturally accounts for
//!  multiple densities for voxels with the same material (not all voxels have same mass).
//!  Electrons are not transported in MC-GPU and therefore we are approximating
//!  that the dose is equal to the KERMA (energy released by the photons alone).
//!  This approximation is acceptable when there is electronic equilibrium and 
//!  when the range of the secondary electrons is shorter than the organ size. 
//!
//!  The function uses atomic functions for a thread-safe access to the GPU memory.
//!  We can check if this tally was disabled in the input file checking if the array
//!  materials_dose was allocated in the GPU (disabled if pointer = NULL).
//!
//!
//!       @param[in] Edep   Energy deposited in the interaction
//!       @param[in] material   Current material id number
//!       @param[out] materials_dose   ulonglong2 array storing the mateials dose [in eV/g] and dose^2 (ie, uncertainty).
////////////////////////////////////////////////////////////////////////////////
#ifdef USING_CUDA
__device__
#endif
inline 
void tally_materials_dose(float* Edep, int* material, ulonglong2* materials_dose)
{
      
// !!DeBuG!! The energy can be tallied directly with atomicAdd in global memory or using shared memory first and then global for whole block if too slow. With the initial testing it looks like using global memory is already very fast!

// !!DeBuG!! WARNING: with many histories and few materials the materials_dose integer variables may overflow!! Using double precision floats would be better. Single precision is not good enough because adding small energies to a large counter would give problems.

#ifdef USING_CUDA
  atomicAdd(&materials_dose[*material].x, __float2ull_rn((*Edep)*SCALE_eV) );  // Energy deposited at the material, scaled by the factor SCALE_eV and rounded.
  atomicAdd(&materials_dose[*material].y, __float2ull_rn((*Edep)*(*Edep)) );   // Square of the dose to estimate standard deviation (not using SCALE_eV for std_dev to prevent overflow)
#else
  materials_dose[*material].x += (unsigned long long int)((*Edep)*SCALE_eV + 0.5f);
  materials_dose[*material].y += (unsigned long long int)((*Edep)*(*Edep) + 0.5f);
#endif     
          
  return;
}

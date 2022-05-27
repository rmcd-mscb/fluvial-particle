=============================
Memory Usage Guide
=============================

Memory scaling
-------------------

The peak memory usage during a *fluvial-particle* simulation scales according to the number of simulated particles and the size of the river meshes. Use the rates below to predict an estimate for the peak memory usage of a simulation (per core), then add on ~0.2 GiB for background processes. For example, simulating 10\ :sup:`6` particles in a 3D mesh with 5*10\ :sup:`6` cells with would be expected to use:  (1.1*10\ :sup:`-7` GiB per cell) * (5*10\ :sup:`6` cells) + (3.7*10\ :sup:`-7` GiB per particle) * (10\ :sup:`6` particles) + 0.2 = 1.1 GiB

Peak memory usage increases with the number of simulated particles at a rate of 3.7 * 10\ :sup:`-7` GiB (0.397 KiB) per particle. This was determined from serial simulations on the 3D Kootenai River ranging from 10\ :sup:`3` to 10\ :sup:`7` passive particles.

.. image:: data/memory_particles.png


In a 2D simulation with one simulated particle, the peak memory usage increases with the number of 2D mesh cells at a rate of 1.5 * 10\ :sup:`-7` GiB (0.161 KiB) per cell.

.. image:: data/memory_2dmesh.png


In a 3D simulation with one simulated particle, the peak memory usage increases with the number of 3D mesh cells at a rate of 1.1 * 10\ :sup:`-7` GiB (0.118 KiB) per cell. 3D simulations also use the same 2D meshes as the 2D simulations -- this rate includes the effects of the 2D mesh.

.. image:: data/memory_3dmesh.png


Output file sizes
-------------------

In addition to scaling with the number of mesh cells and the *total* number of particles (i.e. summed across all cores), the output HDF5 files also scale with the number of printing steps. The particles HDF5 file writes 12 1D arrays per printing time step, each of which store 8-byte numbers. The expected scaling relation of the file size :math:`S` (in bytes) with the total number of printing steps is therefore: 

.. math::
 S_{particles} = N_{particles}*12*8*N_{prints}

The cells HDF5 file writes three datasets per printing time step, one for each of the 1D, 2D, and 3D meshes. The expected relation is:

.. math:: 
 S_{cells} = (N_{sc} + N_{sc}*N_{nc} + N_{sc}*N_{nc}*N_{zc})*4*N_{prints}


As an example, the plots below show the scaling relations for both files from a 3D simulation of 2\ :sup:`27`\ passive particles on the Kootenai River with mesh dimensions :math:`N_{sc}=3008, N_{nc}=80, N_{zc}=15`. The expected scaling relations are thus :math:`S_{particles} = 12*N_{prints}` GiB, :math:`S_{cells} = 0.014*N_{prints}` GiB, rates which match those observed from the simulation.

.. image:: data/memory_particleshdf5.png


.. image:: data/memory_cellshdf5.png
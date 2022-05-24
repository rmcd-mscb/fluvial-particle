======================
Output HDF5 files
======================

Model output is written to two HDF5 files: particles.h5 and cells.h5. The particles.h5 file can be written to independently in parallel during an MPI-enabled run, while the cells.h5 file is written by a single processor in a post-processing loop using the data in particles.h5. Both files contain output from the same time step values. The files are organized similarly, with a single-level of groups in the root folder that hold either grid-related data or data defined over the grids. These data can be visualized with Paraview's XDMF Reader by reading the .xmf files that are also written during model execution.


particles.h5 file organization
---------------------------------

All of the datasets in particles.h5 are stored as NumPy 8-byte floating point numbers (i.e. np.float64 types) with the exception of cellidx2d and cellidx3d which are np.int64 data types.
From the HDF5 root directory, the *particles.h5* file is organized as:

.. image:: data/particles_tree.PNG


.. csv-table:: particles.h5 dataset descriptions
 :file: data/particles.csv
 :widths: 15, 20, 15, 50
 :header-rows: 1


cells.h5 file organization
---------------------------------
All of the datasets in cells.h5 are stored as NumPy 8-byte floating point numbers, i.e. np.float64 types. The fractional particle count data sets only store a single time slice each for reasons related to visualziation with XDMF. 
From the HDF5 root directory, the *cells.h5* file is organized as:

.. image:: data/cells_tree.PNG

.. csv-table:: cells.h5 dataset descriptions
 :file: data/cells.csv
 :widths: 15, 20, 15, 50
 :header-rows: 1
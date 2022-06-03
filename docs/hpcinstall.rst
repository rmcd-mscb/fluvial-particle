========================
Install on Denali HPC
========================

.. role:: bash(code)
 :language: bash


To run *fluvial-particle* with hundreds to thousands of cores on the USGS Denali supercomputer, special instructions need to be followed so that the Python packages reference pre-installed Denali modules. The following steps will install the mpi4py, h5py, and vtk packages with reference to the default MPI module (cray-mpich/7.7.11) and the parallel HDF5 module (cray-hdf5-parallel/1.10.5.2) on Denali.

The available Python module on Denali is version 3.7, but *fluvial-particle* requires version 3.9. Therefore, we will use a conda environment to install the required version of Python and we will install the other required packages with this environment activated.

It is recommended that the Miniconda environment and the *fluvial-particle* git repository are installed into subdirectories of your Caldera project folder. This enables faster loading of all of the Python code and packages during a simulation. Loading from subdirectories of the home directory is slower and can cause a system timeout during initialization.

1. The first step is to prepare the install environment by loading/unloading modules. **IMPORTANT**: this step must be repeated anytime you log out and back in to Denali during this installation process.

    .. code:: bash

     module unload craype-hugepages2M
     module swap PrgEnv-cray/6.0.5 PrgEnv-gnu
     module load cray-hdf5-parallel/1.10.5.2
     export OMP_NUM_THREADS=1

2. Download Miniconda and install

    .. code:: bash

     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash ./Miniconda3-latest-Linux-x86_64.sh

   Be sure to specify a location on your Caldera path when asked where to install conda, e.g. :bash:`/caldera/path/to/your/directory/miniconda3`
   
   When asked by the installer if you would like it to initialize Miniconda3, answer "no". Once installation completes, run :bash:`/path/to/miniconda3/bin/conda init` to initialize. You may need to log out and back in -- remember to repeat step 1!
 

3. Create the conda environment with Python 3.9, then activate using source (this is equivalent to running conda activate fluvial-mpi but is more robust).

    .. code:: bash
     
     conda create -n fluvial-mpi python=3.9
     source /caldera/path/to/miniconda3/bin/activate fluvial-mpi

4. Download and install mpi4py using the system compiler -- this will install mpi4py with reference to the already loaded MPICH module.

    .. code:: bash

     git clone https://github.com/mpi4py/mpi4py.git
     cd mpi4py
     python setup.py build --mpicc="$(which cc) -shared"
     pyton setup.py install
     cd ..

   You can check the mpi4py was installed correctly with the following lines, where <path to python installs> is the path to the site-packages directory given by the first command.

    .. code:: bash

     pip show mpi4py
     ldd <path to python installs>/mpi4py/MPI.cpython*.so

   You should see a line referenching the system MPICH module like :bash:`libmpich_gnu_82.so.3 => /opt/cray/pe/lib64/libmpich_gnu_82.so.3 (0x00007f983c41b000)`.

5. Install h5py with pip, again with reference to the MPICH module and the parallel HDF5 module.

    .. code:: bash

     export CC=cc
     export HDF5_MPI="ON"
     export HDF5_DIR=$HDF5_DIR
     pip install --no-binary=h5py h5py

6. Install vtk

    .. code:: bash

     pip install vtk

7. Install *fluvial-particle* on the fluvial-mpi conda environment. If you haven't already, clone the repository into your Caldera space.

    .. code:: bash

     cd path/to/caldera/fluvparticle
     pip install -e .


And that is it! Your are ready to run massively in parallel on Denali.

An example SLURM submission script follows that runs on a single compute node with 40 cores. Note that the environment commands of step 1 are duplicated in the script. This will run using the settings in user_options.py and will simulate a total of 40*NumPart particles.

.. code-block:: slurm

 #!/bin/bash

 #SBATCH --job-name=fluvparticle-mpi-sim
 #SBATCH -N 1
 #SBATCH -n 40
 #SBATCH -p workq
 #SBATCH --account=<your account>
 #SBATCH --hint=nomultithread
 #SBATCH --time=2-00:00:00
 #SBATCH -o %j.out
 #SBATCH --exclusive

 module unload craype-hugepages2M
 module swap PrgEnv-cray/6.0.5 PrgEnv-gnu
 module load cray-hdf5-parallel/1.10.5.2
 export OMP_NUM_THREADS=1

 source /caldera/projects/css/sas/arc/aprescott/miniconda3/bin/activate fluvial-mpi
 
 srun fluvial_particle_mpi user_options.py .

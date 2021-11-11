"""Top-level package for pygeoapi plugin: Fluvial Particle."""

__author__ = "Richard McDonald"
__email__ = "rmcd@usgs.gov"
__version__ = "0.0.1-dev0"


"""ParticleTrack."""
from datetime import timedelta
import os
from os import getcwd, getpid
import pathlib
import time
import numpy as np
import argparse
from .settings import settings
from .FallingParticles import FallingParticles
from .LarvalParticles import LarvalParticles
from .Particles import Particles
from .RiverGrid import RiverGrid

def checkCommandArguments():
    """Check the users command line arguments. """
    Parser = argparse.ArgumentParser(description="fluvial_particle", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Parser.add_argument('settings_file', help='User settings file')
    Parser.add_argument('output_directory', help='Output directory for results')
    Parser.add_argument('--seed', dest='seed', type=int, default=None, help='Specify a single integer to fix the seed of the random number generator. Only used in serial mode.')

    args = Parser.parse_args()

    return args.settings_file, args.output_directory, args.seed


def get_prng(timer, seed=None):
    """Generate a random seed using time and the process id

    Returns
    -------
    seed : int
        The seed on each core

    """
    if seed is None:
        seed = np.int64(np.abs(((timer()*181)*((getpid()-83)*359))%104729))

    print('Using seed {}'.format(seed), flush=True)

    prng = np.random.RandomState(seed)
    return prng

def simulate(settings, output_directory, timer, seed=None, comm=None):

    t0 = timer()

    # Get rank, number of processors, global number of particles, and local slice indices
    rank = 0
    size = 1
    if not comm is None:
        rank = comm.Get_rank()
        size = comm.Get_size()

    master = rank == 0

    if master:
        print("Beginning simulation", flush=True)

    # Some Variables
    EndTime = settings['SimTime']
    dt = settings['dt']
    min_depth = settings['min_depth']

    lev = settings['LEV']  # lateral eddy viscosity

    beta_x = settings['beta_x']
    beta_y = settings['beta_y']
    beta_z = settings['beta_z']

    # 2D or 3D particle tracking
    Track3D = settings['Track3D']
    print_inc_time = settings['PrintAtTick']

    # Fractional depth that bounds vertical particle positions from bed and WSE
    alpha = 0.5
    if Track3D:
        alpha = 0.01

    # The source file
    file_name_2da = settings['file_name_2da']
    file_name_3da = settings['file_name_3da']

    # Initialize RiverGrid object
    River = RiverGrid(Track3D, file_name_2da, file_name_3da)
    nsc = River.nsc
    num3dcells = River.vtksgrid3d.GetNumberOfCells()
    num2dcells = River.vtksgrid2d.GetNumberOfCells()

    # Initialize particles with initial location and attach RiverGrid
    # npart = 300  # number of particles per processor
    npart = settings['NumPart']

    globalnparts = npart * size  # total number of particles across processors
    start = rank * npart  # slice starting index for HDF5 file
    end = start + npart  # slice ending index (non-inclusive) for HDF5 file

    if master:
        if size > 1:
            s = "Simulating {} particles, {} on each rank".format(globalnparts, npart)
        else:
            s = "Simulating {} particles".format(globalnparts)
        print(s, flush=True)

    xstart, ystart, zstart = settings['StartLoc']
    x = np.full(npart, fill_value=xstart, dtype=np.float64)
    y = np.full(npart, fill_value=ystart, dtype=np.float64)
    z = np.full(npart, fill_value=zstart, dtype=np.float64)

    rng = get_prng(timer, seed)
    
    # Sinusoid properties for larval drift subclass
    amplitude = settings['amplitude']
    period = settings['period']
    min_elev = settings['min_elev']
    ttime = rng.uniform(0.0, period, npart)
    # """ particles = LarvalParticles(
    #     npart, x, y, z, rng, River, Track3D, 0.2, period, min_elev, ttime
    # ) """

    particles = FallingParticles(npart, x, y, z, rng, River, radius=0.000001)
    particles.initialize_location(0.5)  # 0.5 is midpoint of water column

    times = np.arange(dt, EndTime + dt, dt)
    n_times = times.size

    print_inc = np.max([np.int32(print_inc_time / dt), 1])  # smallest possible increment = 1
    print_inc = np.min([print_inc, n_times])  # prevent print increments longer than the simulation
    print_times = times[print_inc - 1:n_times:print_inc]
    # Add final time to print_times, if necessary
    if print_times[-1] != times[-1]:
        print_times = np.append(print_times, times[-1])
    n_prints = print_times.size + 1  # plus one so we can write t=0.0 to file

    # Create HDF5 particles dataset; collective in MPI
    parts_h5 = particles.create_hdf(n_prints, globalnparts, fname=output_directory+'//particles.h5', comm=comm)  # MPI version

    if not comm is None:
        comm.Barrier()

    # Write initial conditions to file
    particles.write_hdf5(parts_h5, 0, start, end, 0.0, rank)

    if not comm is None:
        comm.Barrier()

    t0 = timer()

    for i in range(n_times):  # noqa C901
        # Generate new random numbers
        particles.gen_rands()
        # Interpolate RiverGrid field data to particles
        particles.interp_fields()
        # Calculate dispersion terms
        particles.calc_diffusion_coefs(lev, beta_x, beta_y, beta_z)
        # Move particles (checks on new position done internally)
        particles.move_all(alpha, min_depth, times[i], dt)
        # Check that there are still active particles
        if particles.mask is not None:
            if ~np.any(particles.mask):
                if comm is None:
                    print(f"No active particles remain; exiting loop at time T={times[i]}")
                else:
                    print(f"No active particles remain on processor {rank}; exiting local loop at T={times[i]}")
                break

        # Write to HDF5
        if times[i] in print_times:
            particles.write_hdf5(parts_h5, np.int32(i / print_inc) + 1, start, end, times[i], rank)

            if master:
                e = timer() - t0
                elapsed = str(timedelta(seconds=e))

                if i == 0:
                    print("Remaining time steps {}/{} || Elapsed Time: {} h:m:s".format(n_times-i-1, n_times, elapsed), flush=True)
                else:
                    time_per_time = np.float64(e / i)
                    eta = str(timedelta(seconds=((n_times - i)*time_per_time)))
                    print("Remaining time steps {}/{} || Elapsed Time: {} h:m:s || ETA {} h:m:s".format(n_times-i-1, n_times, elapsed, eta), flush=True)

    if not comm is None:
        comm.Barrier()

    if master:
        print("Finished simulation in {} h:m:s".format(str(timedelta(seconds=timer()-t0)), flush=True))

    # Write xml files and cumulative cell counters
    if master:
        # Create and open xdmf files
        cells1d_xmf = open(output_directory + "//cells_onedim.xmf", "w")
        cells2d_xmf = open(output_directory + "//cells_twodim.xmf", "w")
        cells3d_xmf = open(output_directory + "//cells_threedim.xmf", "w")
        parts_xmf = open(output_directory + "//particles.xmf", "w")
        River.write_hdf5_xmf_header1d(cells1d_xmf)
        River.write_hdf5_xmf_header2d(cells2d_xmf)
        River.write_hdf5_xmf_header3d(cells3d_xmf)
        particles.write_hdf5_xmf_header(parts_xmf)

        # Create cells HDF5 file
        grpc = parts_h5["coordinates"]
        grpp = parts_h5["properties"]
        time = grpc["time"]
        cells_h5 = River.create_hdf5(n_prints, time, output_directory + "//cells.h5")

        NumPartInCell = np.zeros(num2dcells, dtype=np.int64)
        NumPartIn3DCell = np.zeros(num3dcells, dtype=np.int64)
        PartTimeInCell = np.zeros(num2dcells)
        TotPartInCell = np.zeros(num2dcells, dtype=np.int64)
        PartInNSCellPTime = np.zeros(nsc, dtype=np.int64)

        # For every printing time loop, we load the particles data, sum the cell-centered counter arrays,
        # write the arrays to the cells HDF5, and write metadata to the XDMF files
        gen = [t for t in time if not np.isnan(t)]
        for i in range(len(gen)):
            t = gen[i]
            cell2d = grpp["cellidx2d"][i, :]
            cell3d = grpp["cellidx3d"][i, :]
            t = t.item(0)  # this returns a python scalar, for use in f-strings
            particles.write_hdf5_xmf(parts_xmf, t, n_prints, globalnparts, i)

            PartInNSCellPTime[:] = 0
            NumPartIn3DCell[:] = 0
            NumPartInCell[:] = 0
            CI_IDB = cell2d % nsc
            np.add.at(PartInNSCellPTime, CI_IDB, 1)
            np.add.at(PartTimeInCell, cell2d, dt)
            np.add.at(TotPartInCell, cell2d, 1)
            np.add.at(NumPartInCell, cell2d, 1)
            np.add.at(NumPartIn3DCell, cell3d, 1)

            # dims, name, and attrname must be passed to write_hdf5_xmf as iterable objects
            # dtypes too, but it is optional (defaults to "Float")
            name = [[]]
            attrname = [[]]
            name[0] = f"/cells1d/fpc{i}"
            attrname[0] = "FractionalParticleCount"
            data = PartInNSCellPTime / globalnparts
            River.write_hdf5(cells_h5, name[0], data)
            dims = (River.ns - 1,)
            River.write_hdf5_xmf(cells1d_xmf, t, dims, name, attrname, center="Node")

            name = [[], []]
            attrname = [[], []]
            dtypes = [[], []]
            name[0] = f"/cells2d/fpc{i}"
            attrname[0] = "FractionalParticleCount"
            dtypes[0] = "Float"
            dims = (River.ns - 1, River.nn - 1)
            data = (NumPartInCell / globalnparts).reshape(dims)
            River.write_hdf5(cells_h5, name[0], data)
            name[1] = f"/cells2d/tpc{i}"
            attrname[1] = "TotalParticleCount"
            dtypes[1] = "Int"
            dims = (River.ns - 1, River.nn - 1)
            data = TotPartInCell.reshape(dims)
            River.write_hdf5(cells_h5, name[1], data)
            River.write_hdf5_xmf(cells2d_xmf, t, dims, name, attrname, dtypes)

            name = [[]]
            attrname = [[]]
            name[0] = f"/cells3d/fpc{i}"
            attrname[0] = "FractionalParticleCount"
            dims = (River.ns - 1, River.nn - 1, River.nz - 1)
            data = (NumPartIn3DCell / globalnparts).reshape(dims)
            River.write_hdf5(cells_h5, name[0], data)
            River.write_hdf5_xmf(cells3d_xmf, t, dims, name, attrname)

        # Finalize xmf file writing
        River.write_hdf5_xmf_footer(cells1d_xmf)
        River.write_hdf5_xmf_footer(cells2d_xmf)
        River.write_hdf5_xmf_footer(cells3d_xmf)
        particles.write_hdf5_xmf_footer(parts_xmf)
        cells1d_xmf.close()
        cells2d_xmf.close()
        cells3d_xmf.close()
        parts_xmf.close()
        cells_h5.close()
    # end ROOT section
    # the preceeding section could be done on several processors, split over the first index (time)

    # COLLECTIVE file close
    parts_h5.close()

    if master:
        print("Finished in {} h:m:s".format(str(timedelta(seconds=timer()-t0)), flush=True))


def track_serial():

    settings_file, output_directory, seed = checkCommandArguments()
    # sys.path.append(getcwd())

    inputfile = pathlib.Path(settings_file)
    if not inputfile.exists():
        raise Exception(f"Cannot find settings file {inputfile}")
    outdir = pathlib.Path(output_directory)
    if not outdir.is_dir():
        raise Exception(f"Output directory {outdir} does not exist")

    options = settings.read(settings_file)

    simulate(options, output_directory, timer=time.time, seed=seed)

def track_mpi():

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    settings_file, output_directory, seed = checkCommandArguments()
    # sys.path.append(getcwd())

    options = settings.read(settings_file)

    simulate(options, output_directory, comm=comm, timer=MPI.Wtime)
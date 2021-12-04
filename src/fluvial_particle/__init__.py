"""Top-level package for pygeoapi plugin: Fluvial Particle."""

__author__ = "Richard McDonald"
__email__ = "rmcd@usgs.gov"
__version__ = "0.0.1-dev0"


from datetime import timedelta
from os import getpid
import pathlib
import time
import numpy as np
import argparse
import h5py
from .Settings import Settings
from .FallingParticles import FallingParticles  # noqa
from .LarvalParticles import LarvalParticles  # noqa
from .Particles import Particles  # noqa
from .RiverGrid import RiverGrid


def checkcommandarguments():
    """Check the users command line arguments."""
    parser = argparse.ArgumentParser(
        description="fluvial_particle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("settings_file", help="User settings file")
    parser.add_argument("output_directory", help="Output directory for results")
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Specify a single integer to fix the seed of the random number generator. Only used in serial mode.",
    )
    parser.add_argument("--no-postprocess", "--no_postprocess", action="store_false")
    # note: argparse will convert to key="no_postprocess"

    argdict = vars(parser.parse_args())

    inputfile = pathlib.Path(argdict["settings_file"])
    if not inputfile.exists():
        raise Exception(f"Cannot find settings file {inputfile}")
    outdir = pathlib.Path(argdict["output_directory"])
    if not outdir.is_dir():
        raise Exception(f"Output directory {outdir} does not exist")

    return argdict


def get_prng(timer, seed=None):
    """Generate a random seed using time and the process id.

    Returns
    -------
    seed : int
        The seed on each core

    """
    if seed is None:
        seed = np.int64(np.abs(((timer() * 181) * ((getpid() - 83) * 359)) % 104729))

    print(f"Using seed {seed}", flush=True)

    prng = np.random.RandomState(seed)
    return prng


def postprocess(output_directory, river, particles, parts_h5, n_prints, globalnparts):
    """Write xml files and cumulative cell counters.

    Args:
        output_directory ([type]): [description]
        river ([type]): [description]
        particles ([type]): [description]
        parts_h5 ([type]): [description]
        n_prints ([type]): [description]
        globalnparts ([type]): [description]
    """
    # Create and open xdmf files
    cells1d_xmf = open(output_directory + "//cells_onedim.xmf", "w")
    cells2d_xmf = open(output_directory + "//cells_twodim.xmf", "w")
    parts_xmf = open(output_directory + "//particles.xmf", "w")
    river.write_hdf5_xmf_header1d(cells1d_xmf)
    river.write_hdf5_xmf_header2d(cells2d_xmf)
    particles.write_hdf5_xmf_header(parts_xmf)

    # Create cells HDF5 file and arrays
    grpc = parts_h5["coordinates"]
    grpp = parts_h5["properties"]
    time = grpc["time"]
    nsc = river.nsc
    num2dcells = river.vtksgrid2d.GetNumberOfCells()
    cells_h5 = river.create_hdf5(n_prints, time, output_directory + "//cells.h5")
    numpartin2dcell = np.zeros(num2dcells, dtype=np.int64)
    totpartincell = np.zeros(num2dcells, dtype=np.int64)
    numpartin1dcell = np.zeros(nsc, dtype=np.int64)

    if river.track3d:
        num3dcells = river.vtksgrid3d.GetNumberOfCells()
        cells3d_xmf = open(output_directory + "//cells_threedim.xmf", "w")
        river.write_hdf5_xmf_header3d(cells3d_xmf)
        numpartin3dcell = np.zeros(num3dcells, dtype=np.int64)

    # For every printing time loop, we load the particles data, sum the cell-centered counter arrays,
    # write the arrays to the cells HDF5, and write metadata to the XDMF files
    gen = [t for t in time if not np.isnan(t)]
    for i in range(len(gen)):
        t = gen[i].item(0)  # this returns a python scalar, for use in f-strings
        particles.write_hdf5_xmf(parts_xmf, t, n_prints, globalnparts, i)

        cell2d = grpp["cellidx2d"][i, :]
        numpartin1dcell[:] = 0
        numpartin2dcell[:] = 0
        np.add.at(numpartin1dcell, cell2d[cell2d >= 0] % nsc, 1)
        np.add.at(totpartincell, cell2d[cell2d >= 0], 1)
        np.add.at(numpartin2dcell, cell2d[cell2d >= 0], 1)
        if river.track3d:
            cell3d = grpp["cellidx3d"][i, :]
            numpartin3dcell[:] = 0
            np.add.at(numpartin3dcell, cell3d[cell3d >= 0], 1)

        # dims, name, and attrname must be passed to write_hdf5_xmf as iterable objects
        # dtypes too, but it is optional (defaults to "Float")
        name = [[]]
        attrname = [[]]
        name[0] = f"/cells1d/fpc{i}"
        attrname[0] = "FractionalParticleCount"
        data = numpartin1dcell / globalnparts
        river.write_hdf5(cells_h5, name[0], data)
        dims = (river.ns - 1,)
        river.write_hdf5_xmf(cells1d_xmf, t, dims, name, attrname, center="Node")

        name = [[], []]
        attrname = [[], []]
        dtypes = [[], []]
        name[0] = f"/cells2d/fpc{i}"
        attrname[0] = "FractionalParticleCount"
        dtypes[0] = "Float"
        dims = (river.ns - 1, river.nn - 1)
        data = (numpartin2dcell / globalnparts).reshape(dims)
        river.write_hdf5(cells_h5, name[0], data)
        name[1] = f"/cells2d/tpc{i}"
        attrname[1] = "TotalParticleCount"
        dtypes[1] = "Int"
        dims = (river.ns - 1, river.nn - 1)
        data = totpartincell.reshape(dims)
        river.write_hdf5(cells_h5, name[1], data)
        river.write_hdf5_xmf(cells2d_xmf, t, dims, name, attrname, dtypes)

        if river.track3d:
            name = [[]]
            attrname = [[]]
            name[0] = f"/cells3d/fpc{i}"
            attrname[0] = "FractionalParticleCount"
            dims = (river.ns - 1, river.nn - 1, river.nz - 1)
            data = (numpartin3dcell / globalnparts).reshape(dims)
            river.write_hdf5(cells_h5, name[0], data)
            river.write_hdf5_xmf(cells3d_xmf, t, dims, name, attrname)

    # Finalize xmf file writing
    river.write_hdf5_xmf_footer(cells1d_xmf)
    river.write_hdf5_xmf_footer(cells2d_xmf)
    if river.track3d:
        river.write_hdf5_xmf_footer(cells3d_xmf)
        cells3d_xmf.close()
    particles.write_hdf5_xmf_footer(parts_xmf)
    cells1d_xmf.close()
    cells2d_xmf.close()
    parts_xmf.close()
    cells_h5.close()


def simulate(settings, argvars, timer, comm=None):  # noqa
    """Run the fluvial particle simulation.

    Args:
        settings (dict subclass): parameter settings for the simulation
        argvars (dict): dictionary holding command line argument variables
        timer (time object): does timing
        comm (MPI intracomm object): for parallel runs only, MPI communicator
    """
    t0 = timer()

    # Get rank, number of processors, global number of particles, and local slice indices
    rank = 0
    size = 1
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()

    master = rank == 0

    if master:
        print("Beginning simulation", flush=True)

    # Command-line arguments
    output_directory = argvars["output_directory"]
    seed = argvars["seed"]
    postprocessflg = argvars["no_postprocess"]

    # Some Variables
    endtime = settings["SimTime"]
    dt = settings["dt"]
    min_depth = settings["min_depth"]

    lev = settings["LEV"]  # lateral eddy viscosity

    beta_x = settings["beta_x"]
    beta_y = settings["beta_y"]
    beta_z = settings["beta_z"]
    beta = [beta_x, beta_y, beta_z]

    # 2D or 3D particle tracking
    track3d = settings["Track3D"]
    print_inc_time = settings["PrintAtTick"]

    # Fractional depth that bounds vertical particle positions from bed and WSE
    alpha = 0.5
    if track3d:
        alpha = 0.01

    # The source file
    file_name_2da = settings["file_name_2da"]
    file_name_3da = settings["file_name_3da"]

    # Initialize RiverGrid object
    river = RiverGrid(track3d, file_name_2da, file_name_3da)

    # Initialize particles with initial location and attach RiverGrid
    npart = settings["NumPart"]

    globalnparts = npart * size  # total number of particles across processors
    start = rank * npart  # slice starting index for HDF5 file
    end = start + npart  # slice ending index (non-inclusive) for HDF5 file

    if master:
        if size > 1:
            s = "Simulating {} particles, {} on each rank".format(globalnparts, npart)
        else:
            s = "Simulating {} particles".format(globalnparts)
        print(s, flush=True)

    xstart, ystart, zstart = settings["StartLoc"]
    x = np.full(npart, fill_value=xstart, dtype=np.float64)
    y = np.full(npart, fill_value=ystart, dtype=np.float64)
    z = np.full(npart, fill_value=zstart, dtype=np.float64)

    rng = get_prng(timer, seed)

    """ # Sinusoid properties for larval drift subclass
    amplitude = settings["amplitude"]
    period = settings["period"]
    min_elev = settings["min_elev"]
    ttime = rng.uniform(0.0, period, npart)
    particles = LarvalParticles(
        npart, x, y, z, rng, river, track3d, 0.2, period, min_elev, ttime
    ) """

    particles = Particles(
        npart, x, y, z, rng, river, track3d, lev=lev, beta=beta, comm=comm
    )
    # particles = FallingParticles(npart, x, y, z, rng, river, track3d, radius=0.000001)
    particles.initialize_location(0.5)  # 0.5 is midpoint of water column

    # Calc simulation and printing times
    times = np.arange(dt, endtime + dt, dt)
    n_times = times.size
    print_inc = np.max([np.int32(print_inc_time / dt), 1])  # bound below
    print_inc = np.min([print_inc, n_times])  # bound above
    print_times = times[print_inc - 1 : n_times : print_inc]
    # Add final time to print_times, if necessary
    if print_times[-1] != times[-1]:
        print_times = np.append(print_times, times[-1])
    n_prints = print_times.size + 1  # plus one so we can write t=0 to file

    # Create HDF5 particles dataset; collective in MPI
    fname = output_directory + "//particles.h5"
    parts_h5 = particles.create_hdf5(n_prints, globalnparts, fname=fname, comm=comm)

    if comm is not None:
        comm.Barrier()

    # Write initial conditions to file
    particles.write_hdf5(parts_h5, 0, start, end, 0.0, rank)

    if comm is not None:
        comm.Barrier()

    t0 = timer()

    for i in range(n_times):  # noqa C901
        # Move particles
        particles.move(alpha, min_depth, times[i], dt)

        # Check that there are still active particles
        if particles.mask is not None:
            if ~np.any(particles.mask):
                if comm is None:
                    print(
                        f"No active particles remain; exiting loop at time T={times[i]}",
                        flush=True,
                    )
                else:
                    print(
                        f"No active particles remain on processor {rank}; exiting local loop at T={times[i]}",
                        flush=True,
                    )
                break

        # Write to HDF5
        tidx = np.searchsorted(print_times, times[i])
        if print_times[tidx] == times[i]:
            particles.write_hdf5(
                parts_h5, np.int32(i / print_inc) + 1, start, end, times[i], rank
            )

            if master:
                e = timer() - t0
                elapsed = str(timedelta(seconds=e))

                if i == 0:
                    print(
                        f"Remaining time steps {n_times - i - 1}/{n_times} || Elapsed Time: {elapsed} h:m:s",
                        flush=True,
                    )
                else:
                    time_per_time = np.float64(e / i)
                    eta = str(timedelta(seconds=((n_times - i) * time_per_time)))
                    print(
                        f"Remaining time steps {n_times - i - 1}/{n_times} || Elapsed Time: {elapsed} h:m:s || ETA {eta} h:m:s",
                        flush=True,
                    )

    if comm is not None:
        comm.Barrier()

    # COLLECTIVE file close
    parts_h5.close()

    if master:
        print(
            f"Finished simulation in {str(timedelta(seconds=timer() - t0))} h:m:s",
            flush=True,
        )

    if master and postprocessflg:
        print("Post-processing...")
        parts_h5 = h5py.File(fname, "r")
        postprocess(
            output_directory, river, particles, parts_h5, n_prints, globalnparts
        )
        parts_h5.close()

    if comm is not None:
        comm.Barrier()

    if master:
        print(f"Finished in {str(timedelta(seconds=timer()-t0))} h:m:s", flush=True)


def track_serial():
    """Run fluvial particle in serial."""
    argdict = checkcommandarguments()
    settings_file = argdict["settings_file"]
    options = Settings.read(settings_file)

    simulate(options, argdict, timer=time.time)


def track_mpi():
    """Run fluvial particle in parallel."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    argdict = checkcommandarguments()
    settings_file = argdict["settings_file"]
    seed = argdict["seed"]
    if seed is not None:
        print("Warning: user-input seed ignored in parallel execution mode.")
        argdict["seed"] = None
    options = Settings.read(settings_file)

    simulate(options, argdict, timer=MPI.Wtime, comm=comm)

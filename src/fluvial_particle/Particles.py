"""Particles Class module."""
import h5py
import numpy as np
import vtk


class Particles:
    """A class of particles, each with a velocity, size, and mass."""

    def __init__(self, nparts, x, y, z, rng, mesh, track2d=0, track3d=1):
        """Initialize instance of class Particles.

        Args:
            nparts (int): number of particles in this instance
            x (float): x-coordinate of each particle, numpy array of length nparts
            y (float): y-coordinate of each particle, numpy array of length nparts
            z (float): z-coordinate of each particle, numpy array of length nparts
            rng (Numpy object): random number generator
            mesh (RiverGrid): class instance of the river hydrodynamic data
            track2d (bool): 1 if 2D model run, 0 else
            track3d (bool): 1 if 3D model run, 0 else
        """
        self.nparts = nparts
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.rng = rng
        self.mesh = mesh
        self.track2d = track2d
        self.track3d = track3d
        # Add an XOR on track2d & track3d ?

        self._bedelev = np.zeros(nparts)
        self._wse = np.zeros(nparts)
        self._normdepth = np.full(nparts, 0.5)
        self.indices = np.arange(nparts)
        self._cellindex2d = np.zeros(nparts, dtype=np.int64)
        self._cellindex3d = np.zeros(nparts, dtype=np.int64)
        self._depth = np.zeros(nparts)
        self._velx = np.zeros(nparts)
        self._vely = np.zeros(nparts)
        self._velz = np.zeros(nparts)
        self._shearstress = np.zeros(nparts)
        self._ustar = np.zeros(nparts)
        self._diffx = np.zeros(nparts)
        self._diffy = np.zeros(nparts)
        self._diffz = np.zeros(nparts)
        self._time = np.zeros(nparts)
        self._htabvbed = np.zeros(nparts)
        self.xrnum = np.zeros(nparts)  # make property?
        self.yrnum = np.zeros(nparts)  # make property?
        self.zrnum = np.zeros(nparts)  # make property?

        # Not used:
        # tmpibc = np.zeros(nparts)
        # tmpvel = np.zeros(npart)

    def adjust_z(self, pz, alpha):
        """Check that new particle vertical position is within bounds.

        Args:
            pz ([type]): [description]
            alpha (float): bounds particle in fractional water column to [alpha, 1-alpha]
        """
        # check on alpha? only makes sense for alpha<=0.5
        a = self.indices[pz > self.wse - alpha * self.depth]
        b = self.indices[pz < self.bedelev + alpha * self.depth]
        pz[a] = self.wse[a] - alpha * self.depth[a]
        pz[b] = self.bedelev[b] + alpha * self.depth[b]

    def calc_diffusion_coefs(self, lev, bx, by, bz):
        """Calculate diffusion coefficients.

        Args:
            lev ([type]): [description]
            bx ([type]): [description]
            by ([type]): [description]
            bz ([type]): [description]
        """
        ustarh = self.depth * self.ustar
        self.diffx = lev + bx * ustarh
        self.diffy = lev + by * ustarh
        self.diffz = bz * ustarh  # lev + bz * ustarh

    def create_hdf(self, dimtime, globalnparts, fname="particles.h5"):
        """Create an HDF5 file to write incremental particles results.

        Args:
            dimtime (int): size of first dimension, indexes time slices
            globalnparts (int): global number of particles, distributed across processors
            fname (string): name of the HDF5 file

        Returns:
            [type]: [description]
        """
        parts_h5 = h5py.File(fname, "w")
        # parts_h5 = h5py.File(fname, "w", driver="mpio", comm=MPI.COMM_WORLD)  # MPI version
        grpc = parts_h5.create_group("coordinates")
        grpc.attrs["Description"] = "Position x,y,z of particles at printing time steps"
        grpc.create_dataset("x", (dimtime, globalnparts), dtype="f")
        grpc.create_dataset("y", (dimtime, globalnparts), dtype="f")
        grpc.create_dataset("z", (dimtime, globalnparts), dtype="f")
        grpc.create_dataset("time", (dimtime, 1), dtype="f")

        grpp = parts_h5.create_group("properties")
        grpp.create_dataset("bedelev", (dimtime, globalnparts), dtype="f")
        grpp.create_dataset("cellidx2d", (dimtime, globalnparts), dtype="i")
        grpp.create_dataset("cellidx3d", (dimtime, globalnparts), dtype="i")
        grpp.create_dataset("htabvbed", (dimtime, globalnparts), dtype="f")
        grpp.create_dataset("velvec", (dimtime, globalnparts, 3), dtype="f")
        grpp.create_dataset("wse", (dimtime, globalnparts), dtype="f")
        grpp["bedelev"].attrs[
            "Description"
        ] = "Bed elevation at x,y position of particles"
        grpp["cellidx2d"].attrs[
            "Description"
        ] = "Index of 2D grid cell containing each particle"
        grpp["cellidx3d"].attrs[
            "Description"
        ] = "Index of 3D grid cell containing each particle"
        grpp["htabvbed"].attrs["Description"] = "Height of particle above bed elevation"
        grpp["velvec"].attrs["Description"] = "Velocity vector (u,v,w) of particles"
        grpp["wse"].attrs[
            "Description"
        ] = "Water surface elevation at x,y position of particles"
        return parts_h5

    def find_pos_in_2dcell(self, newpoint2d, cellid):
        """Find position in 2D cell, return info for interpolation.

        Args:
            newpoint2d ([type]): [description]
            cellid ([type]): [description]

        Returns:
            [type]: [description]
        """
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        clspoint = [0.0, 0.0, 0.0]
        tmpid = vtk.mutable(0)
        vtkid2 = vtk.mutable(0)
        vtkcell2d = self.mesh.vtksgrid2d.GetCell(cellid)
        tmpres = vtkcell2d.EvaluatePosition(  # noqa F841
            newpoint2d, clspoint, tmpid, pcoords, vtkid2, weights
        )
        numpts = vtkcell2d.GetNumberOfPoints()
        idlist1 = vtkcell2d.GetPointIds()
        return weights, idlist1, numpts

    def gen_rands(self):
        """Generate standard normal random numbers."""
        self.xrnum = self.rng.standard_normal(self.nparts)
        self.yrnum = self.rng.standard_normal(self.nparts)
        if self.track3d:
            self.zrnum = self.rng.standard_normal(self.nparts)

    def get_total_position(self):
        """Return complete position of particle."""
        return (
            self.time,
            self.cellindex2d,
            self.x,
            self.y,
            self.z,
            self.bedelev,
            self.htabvbed,
            self.wse,
        )

    def handle_dry_parts(self, px, py, dry, dt):
        """Adjust trajectories of dry particles.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            dry ([type]): [description]
            dt ([type]): [description]
        """
        # Move dry particles with only 2d random motion
        a = self.indices[dry]
        self.perturb_random_only_2d(px, py, a, dt)
        # Run is_part_wet again
        wet2 = self.is_part_wet(px, py, a)
        if np.any(~wet2):
            b = a[~wet2]
            # Any still dry particles will have no positional update this step
            px[b] = self.x[b]
            py[b] = self.y[b]
            # update cell indices
            for i in np.nditer(b):
                point = [px[i], py[i], 0.0]
                self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)

    def initialize_location(self, frac):
        """Initialize position in water column and interpolate mesh arrays.

        Args:
            frac ([type]): [description]
        """
        # ASSERT check that x,y are within the mesh domain
        # ASSERT check that frac in (epsilon, 1-epsilon)
        self.interp_fields
        self.z = self.bedelev + frac * self.depth

    def interp_cell_value(self, weights, idlist1, numpts, valarray):
        """Interpolate valarray at a point.

        Args:
            weights ([type]): [description]
            idlist1 ([type]): [description]
            numpts ([type]): [description]
            valarray ([type]): [description]

        Returns:
            [type]: [description]
        """
        tmpval = np.float64(0.0)
        for i in range(numpts):
            tmpval += weights[i] * valarray.GetTuple(idlist1.GetId(i))[0]

        return tmpval

    @property
    def interp_field_3d(self):
        """Interpolate 3D velocity field at current particles' positions."""
        idlist1 = vtk.vtkIdList()
        for i in range(self.nparts):
            point = [self.x[i], self.y[i], self.z[i]]
            self.cellindex3d[i] = self.mesh.CellLocator3D.FindCell(point)
            if self.cellindex3d[i] >= 0:
                (
                    result,
                    dist,
                    tmp3dux,
                    tmp3duy,
                    tmp3duz,
                ) = self.interp_vel3d_value(point, self.cellindex3d[i])
                self.velx[i] = tmp3dux
                self.vely[i] = tmp3duy
                self.velz[i] = tmp3duz
            else:
                print("3d findcell failed, particle number: ", i)
                print("switching to FindCellsAlongLine() method")
                pp1 = [point[0], point[1], self.wse[i] + 10]
                pp2 = [point[0], point[1], self.bedelev[i] - 10]
                self.mesh.CellLocator3D.FindCellsAlongLine(pp1, pp2, 0.0, idlist1)
                maxdist = 1e6
                for t in range(idlist1.GetNumberOfIds()):
                    (
                        result,
                        dist,
                        tmp3dux,
                        tmp3duy,
                        tmp3duz,
                    ) = self.interp_vel3d_value(point, idlist1.GetId(t))
                    if result == 1:
                        self.velx[i] = tmp3dux
                        self.vely[i] = tmp3duy
                        self.velz[i] = tmp3duz
                        self.cellindex3d[i] = idlist1.GetId(t)
                        break
                    elif dist < maxdist:
                        maxdist = dist
                        self.velx[i] = tmp3dux
                        self.vely[i] = tmp3duy
                        self.velz[i] = tmp3duz
                        self.cellindex3d[i] = idlist1.GetId(t)
                if self.cellindex3d[i] < 0:
                    print("part still out of 3d grid")
                    self.velx[i] = 0.0
                    self.vely[i] = 0.0
                    self.velz[i] = 0.0
                """print("3d findcell failed, particle number: ", i)
                print("Particle location: ", point3d_2[i, :])
                print("Particle fractional depth: ", normdepth[i])
                print("closest 3D cell, 2Dcell: ", cellid3d[i], cellid[i])
                vtkcell = vtksgrid2d.GetCell(cellid[i])
                vtkptlist = vtkcell.GetPointIds()
                vtkpts = vtkcell.GetPoints()
                print("2D grid points:")
                for j in range(vtkcell.GetNumberOfPoints()):
                print(vtkpts.GetPoint(j))
                print(Elevation_2D.GetTuple(vtkptlist.GetId(j)))
                vtkcell = vtksgrid3d.GetCell(cellid3d[i])
                vtkpts = vtkcell.GetPoints()
                print("3D grid points:")
                for j in range(vtkpts.GetNumberOfPoints()):
                print(vtkpts.GetPoint(j)) """

    @property
    def interp_fields(self):
        """Interpolate mesh fields at current particles' positions."""
        # Find current location in 2D grid and interpolate 2D fields
        for i in range(self.nparts):
            point = [self.x[i], self.y[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
            weights, idlist1, numpts = self.find_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            self.bedelev[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.WSE_2D
            )
            # self.depth[i] = self.interp_cell_value(
            #     weights, idlist1, numpts, self.mesh.Depth_2D
            # )
            # self.tmpibc[i] = self.interp_cell_value(
            #     weights, idlist1, numpts, self.mesh.IBC_2D
            # )
            # tmpvel[i] = interp_cell_value(weights, idlist1, numpts, Velocity_2D)
            self.shearstress[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.ShearStress2D
            )
            if self.track2d:
                self.velx[i], self.vely[i] = self.interp_vel2d_value(
                    weights, idlist1, numpts
                )
        self.depth = self.wse - self.bedelev
        # check shear stress (without error print statements)
        self.shearstress = np.where(self.shearstress < 0.0, 0.0, self.shearstress)
        self.ustar = (self.shearstress / 1000.0) ** 0.5

        if self.track3d:
            self.normdepth = (self.z - self.bedelev) / self.depth
            # Get 3D Velocity Components
            self.interp_field_3d

    def interp_vel2d_value(self, weights, idlist1, numpts):
        """Interpolate 2D velocity vector at a point.

        Args:
            weights ([type]): [description]
            idlist1 ([type]): [description]
            numpts ([type]): [description]

        Returns:
            [type]: [description]
        """
        tmpxval = np.float64(0.0)
        tmpyval = np.float64(0.0)
        for i in range(numpts):
            tmpxval += (
                weights[i] * self.mesh.VelocityVec2D.GetTuple(idlist1.GetId(i))[0]
            )
            tmpyval += (
                weights[i] * self.mesh.VelocityVec2D.GetTuple(idlist1.GetId(i))[1]
            )
        return tmpxval, tmpyval

    def interp_vel3d_value(self, newpoint3d, cellid):
        """Interpolate 3D velocity vector at a point.

        Args:
            newpoint3d ([type]): [description]
            cellid ([type]): [description]

        Returns:
            [type]: [description]
        """
        pcoords = [0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        clspoint = [0.0, 0.0, 0.0]
        tmpid = vtk.mutable(0)
        vtkid2 = vtk.mutable(0)
        vtkcell3d = self.mesh.vtksgrid3d.GetCell(cellid)
        result = vtkcell3d.EvaluatePosition(
            newpoint3d, clspoint, tmpid, pcoords, vtkid2, weights
        )
        numpts = vtkcell3d.GetNumberOfPoints()
        idlist1 = vtkcell3d.GetPointIds()
        tmpxval = np.float64(0.0)
        tmpyval = np.float64(0.0)
        for i in range(0, numpts):
            tmpxval += (
                weights[i] * self.mesh.VelocityVec3D.GetTuple(idlist1.GetId(i))[0]
            )
            tmpyval += (
                weights[i] * self.mesh.VelocityVec3D.GetTuple(idlist1.GetId(i))[1]
            )
        return result, vtkid2, tmpxval, tmpyval, 0.0

    def is_part_wet(self, px, py, a):
        """Determine if particles' new positions are wet.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            a (int): Numpy array of indices to check

        Returns:
            [type]: [description]
        """
        wet = np.empty(np.size(a), dtype=bool)
        j = 0
        for i in np.nditer(a):
            point = [px[i], py[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
            weights, idlist1, numpts = self.find_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            wet[j] = self.is_part_wet_kernel(weights, idlist1, numpts)
            j += 1
        return wet

    def is_part_wet_kernel(self, weights, idlist1, numpts):
        """Interpolate IBC mesh array to a point.

        Args:
            weights ([type]): [description]
            idlist1 ([type]): [description]
            numpts ([type]): [description]

        Returns:
            [type]: [description]
        """
        tmpibc = 0.0
        for i in range(numpts):
            tmpibc += weights[i] * self.mesh.IBC_2D.GetTuple(idlist1.GetId(i))[0]
        if tmpibc >= 0.9999999:
            return True
        else:
            return False

    def move_all(self, alpha, min_depth, time, dt):
        """Update position based on speed, angle.

        Args:
            alpha (float): bounding scalar for adjust_z
            min_depth (float): minimum depth scalar that particles can enter
            time (float): the new time at end of position update
            dt (float): time step
        """
        # first perturb 2d only
        px = np.copy(self.x)
        py = np.copy(self.y)
        self.perturb_2d(px, py, dt)

        # check if new positions are wet
        wet = self.is_part_wet(px, py, self.indices)
        if np.any(~wet):
            self.handle_dry_parts(px, py, ~wet, dt)

        # update bed elevation, wse, depth
        for i in range(self.nparts):
            point = [px[i], py[i], 0.0]
            weights, idlist1, numpts = self.find_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            self.bedelev[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.WSE_2D
            )
        self.depth = self.wse - self.bedelev

        # Prevent particles from entering cells where depth < min_depth
        if np.any(self.depth < min_depth):
            self.prevent_mindepth(px, py, min_depth)

        # Perturb vertical, random wiggle by default
        # subclasses can update perturb_z for an active drift
        pz = self.perturb_z(dt)
        self.adjust_z(pz, alpha)

        # Move particles
        self.x = px
        self.y = py
        self.z = pz

        # Update some info
        self.htabvbed = self.z - self.bedelev
        self.time.fill(time)

    def perturb_2d(self, px, py, dt):
        """Project particles' 2D trajectories.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            dt ([type]): [description]
        """
        vx = self.velx
        vy = self.vely
        velmag = (vx ** 2 + vy ** 2) ** 0.5
        xranwalk = self.xrnum * (2.0 * self.diffx * dt) ** 0.5
        yranwalk = self.yrnum * (2.0 * self.diffy * dt) ** 0.5
        # Move and update positions in-place on each array
        a = self.indices[velmag > 0.0]
        px[a] += (
            vx[a] * dt
            + ((xranwalk[a] * vx[a]) / velmag[a])
            - ((yranwalk[a] * vy[a]) / velmag[a])
        )
        py[a] += (
            vy[a] * dt
            + ((xranwalk[a] * vy[a]) / velmag[a])
            + ((yranwalk[a] * vx[a]) / velmag[a])
        )

    def perturb_random_only_2d(self, px, py, idx, dt):
        """Future 2D position based on random walk only.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            idx ([type]): [description]
            dt ([type]): [description]
        """
        px[idx] = self.x[idx] + self.xrnum[idx] * (2.0 * self.diffx[idx] * dt) ** 0.5
        py[idx] = self.y[idx] + self.yrnum[idx] * (2.0 * self.diffy[idx] * dt) ** 0.5

    def perturb_z(self, dt):
        """Project particles' vertical trajectories, random wiggle.

        Args:
            dt ([type]): [description]

        Returns:
            [type]: [description]
        """
        zranwalk = self.zrnum * (2.0 * self.diffz * dt) ** 0.5
        pz = self.bedelev + (self.normdepth * self.depth) + self.velz * dt + zranwalk
        return pz

    def prevent_mindepth(self, px, py, min_depth):
        """Prevent particles from entering a position with depth < min_depth.

        Args:
            px ([type]): [description]
            py ([type]): [description]
            min_depth ([type]): [description]
        """
        print("particle entered min_depth")
        a = self.indices[self.depth < min_depth]
        # update cell indices and interpolations
        for i in np.nditer(a):
            point = [px[i], py[i], 0.0]
            self.cellindex2d[i] = self.mesh.CellLocator2D.FindCell(point)
            weights, idlist1, numpts = self.find_pos_in_2dcell(
                point, self.cellindex2d[i]
            )
            self.bedelev[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.Elevation_2D
            )
            self.wse[i] = self.interp_cell_value(
                weights, idlist1, numpts, self.mesh.WSE_2D
            )
        self.depth[a] = self.wse[a] - self.bedelev[a]

    def write_hdf5(self, obj, idx, start, end, time, rank):
        """[summary].

        Args:
            obj ([type]): h5py file object created with self.create_hdf()
            idx ([type]): [description]
            start ([type]): [description]
            end ([type]): [description]
            time ([type]): [description]
            rank ([type]): [description]
        """
        grpc = obj["coordinates"]
        grpp = obj["properties"]
        grpc["x"][idx, start:end] = self.x
        grpc["y"][idx, start:end] = self.y
        grpc["z"][idx, start:end] = self.z
        if rank == 0:
            grpc["time"][idx] = time
        grpp["bedelev"][idx, start:end] = self.bedelev
        grpp["htabvbed"][idx, start:end] = self.htabvbed
        grpp["wse"][idx, start:end] = self.wse
        grpp["velvec"][idx, start:end, :] = np.vstack(
            (self.velx, self.vely, self.velz)
        ).T
        grpp["cellidx2d"][idx, start:end] = self.cellindex2d
        grpp["cellidx3d"][idx, start:end] = self.cellindex3d

    def write_hdf5_xmf(self, filexmf, time, dimtime, nparts, idx):
        """[summary].

        Args:
            filexmf ([type]): [description]
            time ([type]): [description]
            dimtime ([type]): [description]
            nparts ([type]): Global number of particles (i.e. summed across processors)
            idx ([type]): [description]
        """
        filexmf.write(
            f"""
            <Grid GridType="Uniform">
                <Time Value="{time}"/>
                <Topology NodesPerElement="{nparts}" TopologyType="Polyvertex"/>
                <Geometry GeometryType="X_Y_Z" Name="particles">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {idx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/coordinates/x
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {idx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/coordinates/y
                        </DataItem>
                    </DataItem>
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                            {idx} 0
                            1 1
                            1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/coordinates/z
                        </DataItem>
                    </DataItem>
                </Geometry>
                <Attribute Name="BedElevation" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {idx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/properties/bedelev
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="CellIndex2D" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {idx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/properties/cellidx2d
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="CellIndex3D" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {idx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/properties/cellidx3d
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="HeightAboveBed" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {idx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/properties/htabvbed
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="WaterSurfaceElevation" AttributeType="Scalar" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts}" Format="XML">
                        <DataItem Dimensions="3 2" Format="XML">
                        {idx} 0
                        1 1
                        1 {nparts}
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts}" Format="HDF">
                            particles.h5:/properties/wse
                        </DataItem>
                    </DataItem>
                </Attribute>
                <Attribute Name="VelocityVector" AttributeType="Vector" Center="Node">
                    <DataItem ItemType="HyperSlab" Dimensions="1 {nparts} 3" Format="XML">
                        <DataItem Dimensions="3 3" Format="XML">
                        {idx} 0 0
                        1 1 1
                        1 {nparts} 3
                        </DataItem>
                        <DataItem Dimensions="{dimtime} {nparts} 3" Format="HDF">
                            particles.h5:/properties/velvec
                        </DataItem>
                    </DataItem>
                </Attribute>
            </Grid>"""
        )

    def write_hdf5_xmf_footer(self, filexmf):
        """[summary].

        Args:
            filexmf ([type]): [description]
        """
        filexmf.write(
            """
                </Grid>
            </Domain>
        </Xdmf>
        """
        )

    def write_hdf5_xmf_header(self, filexmf):
        """[summary].

        Args:
            filexmf ([type]): [description]
        """
        filexmf.write(
            """<Xdmf Version="3.0">
            <Domain>
                <Grid GridType="Collection" CollectionType="Temporal">"""
        )

    # Properties

    @property
    def bedelev(self):
        """Get bedelev.

        Returns:
            [type]: [description]
        """
        return self._bedelev

    @bedelev.setter
    def bedelev(self, values):
        """Set bedelev.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "bedelev.setter wrong size etc. etc."
        )
        self._bedelev = values

    @property
    def cellindex2d(self):
        """Get cellindex2d.

        Returns:
            [type]: [description]
        """
        return self._cellindex2d

    @cellindex2d.setter
    def cellindex2d(self, values):
        """Set cellindex2d.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "cellindex2d.setter wrong size etc. etc."
        )
        self._cellindex2d = values

    @property
    def cellindex3d(self):
        """Get cellindex3d.

        Returns:
            [type]: [description]
        """
        return self._cellindex3d

    @cellindex3d.setter
    def cellindex3d(self, values):
        """Set cellindex3d.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "cellindex3d.setter wrong size etc. etc."
        )
        self._cellindex3d = values

    @property
    def depth(self):
        """Get depth.

        Returns:
            [type]: [description]
        """
        return self._depth

    @depth.setter
    def depth(self, values):
        """Set depth.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "depth.setter wrong size etc. etc."
        )
        self._depth = values

    @property
    def diffx(self):
        """Get diffx.

        Returns:
            [type]: [description]
        """
        return self._diffx

    @diffx.setter
    def diffx(self, values):
        """Set diffx.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "diffx.setter wrong size etc. etc."
        )
        self._diffx = values

    @property
    def diffy(self):
        """Get diffy.

        Returns:
            [type]: [description]
        """
        return self._diffy

    @diffy.setter
    def diffy(self, values):
        """Set diffy.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "diffy.setter wrong size etc. etc."
        )
        self._diffy = values

    @property
    def diffz(self):
        """Get diffz.

        Returns:
            [type]: [description]
        """
        return self._diffz

    @diffz.setter
    def diffz(self, values):
        """Set diffz.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "diffz.setter wrong size etc. etc."
        )
        self._diffz = values

    @property
    def htabvbed(self):
        """Get htabvbed.

        Returns:
            [type]: [description]
        """
        return self._htabvbed

    @htabvbed.setter
    def htabvbed(self, values):
        """Set htabvbed.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "htabvbed.setter wrong size etc. etc."
        )
        self._htabvbed = values

    @property
    def mesh(self):
        """Get mesh.

        Returns:
            [type]: [description]
        """
        return self._mesh

    @mesh.setter
    def mesh(self, values):
        """Set mesh.

        Args:
            values ([type]): [description]
        """
        self._mesh = values

    @property
    def normdepth(self):
        """Get normdepth.

        Returns:
            [type]: [description]
        """
        return self._normdepth

    @normdepth.setter
    def normdepth(self, values):
        """Set normdepth.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "normdepth.setter wrong size etc. etc."
        )
        self._normdepth = values

    @property
    def nparts(self):
        """Get nparts.

        Returns:
            [type]: [description]
        """
        return self._nparts

    @nparts.setter
    def nparts(self, value):
        """Set nparts.

        Args:
            value ([type]): [description]
        """
        assert value > 0, ValueError("# particles < 1")  # noqa: S101
        assert isinstance(value, int), TypeError("nparts must be int")  # noqa: S101
        self._nparts = value

    @property
    def rng(self):
        """Get rng.

        Returns:
            [type]: [description]
        """
        return self._rng

    @rng.setter
    def rng(self, values):
        """Set rng.

        Args:
            values ([type]): [description]
        """
        self._rng = values

    @property
    def shearstress(self):
        """Get shearstress.

        Returns:
            [type]: [description]
        """
        return self._shearstress

    @shearstress.setter
    def shearstress(self, values):
        """Set shearstress.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "shearstress.setter wrong size etc. etc."
        )
        self._shearstress = values

    @property
    def time(self):
        """Get time.

        Returns:
            [type]: [description]
        """
        return self._time

    @time.setter
    def time(self, values):
        """Set time.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "time.setter wrong size etc. etc."
        )
        self._time = values

    @property
    def track2d(self):
        """Get track2d.

        Returns:
            [type]: [description]
        """
        return self._track2d

    @track2d.setter
    def track2d(self, value):
        """Set track2d.

        Args:
            value ([type]): [description]
        """
        assert isinstance(value, int), TypeError("track2d must be int")  # noqa: S101
        assert value >= 0 and value < 2, ValueError(  # noqa: S101
            "track2d must be 0 or 1"
        )
        self._track2d = value

    @property
    def track3d(self):
        """Get track3d.

        Returns:
            [type]: [description]
        """
        return self._track3d

    @track3d.setter
    def track3d(self, value):
        """Set track3d.

        Args:
            value ([type]): [description]
        """
        assert isinstance(value, int), TypeError("track3d must be int")  # noqa: S101
        assert value >= 0 and value < 2, ValueError(  # noqa: S101
            "track3d must be 0 or 1"
        )
        self._track3d = value

    @property
    def ustar(self):
        """Get ustar.

        Returns:
            [type]: [description]
        """
        return self._ustar

    @ustar.setter
    def ustar(self, values):
        """Set ustar.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "ustar.setter wrong size etc. etc."
        )
        self._ustar = values

    @property
    def velx(self):
        """Get velx.

        Returns:
            [type]: [description]
        """
        return self._velx

    @velx.setter
    def velx(self, values):
        """Set velx.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "velx.setter wrong size etc. etc."
        )
        self._velx = values

    @property
    def vely(self):
        """Get vely.

        Returns:
            [type]: [description]
        """
        return self._vely

    @vely.setter
    def vely(self, values):
        """Set vely.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "vely.setter wrong size etc. etc."
        )
        self._vely = values

    @property
    def velz(self):
        """Get velz.

        Returns:
            [type]: [description]
        """
        return self._velz

    @velz.setter
    def velz(self, values):
        """Set velz.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "velz.setter wrong size etc. etc."
        )
        self._velz = values

    @property
    def wse(self):
        """Get wse.

        Returns:
            [type]: [description]
        """
        return self._wse

    @wse.setter
    def wse(self, values):
        """Set wse.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "wse.setter wrong size etc. etc."
        )
        self._wse = values

    @property
    def x(self):
        """Get x.

        Returns:
            [type]: [description]
        """
        return self._x

    @x.setter
    def x(self, values):
        """Set x.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "x.setter wrong size etc. etc."
        )
        self._x = values

    @property
    def y(self):
        """Get y.

        Returns:
            [type]: [description]
        """
        return self._y

    @y.setter
    def y(self, values):
        """Set y.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "y.setter wrong size etc. etc."
        )
        self._y = values

    @property
    def z(self):
        """Get z.

        Returns:
            [type]: [description]
        """
        return self._z

    @z.setter
    def z(self, values):
        """Set z.

        Args:
            values ([type]): [description]
        """
        assert np.size(values) == self.nparts, ValueError(  # noqa: S101
            "z.setter wrong size etc. etc."
        )
        self._z = values

"""PVD (ParaView Data) collection writer for time series output."""

from pathlib import Path


class PVDWriter:
    """Write PVD collection files for time-varying VTK data.

    PVD files are XML files that reference multiple VTK data files
    (VTP, VTS, etc.) with associated timestep information. ParaView
    uses these to load and animate time series data.
    """

    def __init__(self, filepath: Path | str):
        """Initialize the PVD writer.

        Args:
            filepath: Path where the PVD file will be written.
        """
        self.filepath = Path(filepath)
        self.entries: list[tuple[float, str]] = []

    def add_timestep(self, time: float, data_file: Path | str) -> None:
        """Add a timestep entry to the collection.

        Args:
            time: Simulation time for this entry.
            data_file: Path to the data file (VTP, VTS, etc.).
                       Will be stored as relative path from PVD location.
        """
        data_file = Path(data_file)
        # Store relative path from PVD file location
        try:
            rel_path = data_file.relative_to(self.filepath.parent)
        except ValueError:
            # If not relative, use the filename only (assumes same directory)
            rel_path = data_file.name
        rel_path_str = rel_path.as_posix() if isinstance(rel_path, Path) else str(rel_path)
        self.entries.append((time, rel_path_str))

    def write(self) -> None:
        """Write the PVD collection file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with self.filepath.open("w", encoding="utf-8") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">\n')
            f.write("  <Collection>\n")
            for time, datafile in sorted(self.entries, key=lambda x: x[0]):
                f.write(f'    <DataSet timestep="{time}" file="{datafile}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")

    def __len__(self) -> int:
        """Return the number of timesteps in the collection."""
        return len(self.entries)

"""1D river-network particle tracking driven by network hydraulics exports."""

from .config import DispersionConfig, NetworkConfig, get_network_config_template
from .network import Network, NetworkBins
from .provider import FileHydraulicsProvider, HydraulicsProvider
from .results import NetworkResults
from .run import run_network_simulation
from .solver import NetworkSolver
from .sources import ParticleSchedule, estimate_particles, expand_sources


__all__ = [
    "DispersionConfig",
    "FileHydraulicsProvider",
    "HydraulicsProvider",
    "Network",
    "NetworkBins",
    "NetworkConfig",
    "NetworkResults",
    "NetworkSolver",
    "ParticleSchedule",
    "estimate_particles",
    "expand_sources",
    "get_network_config_template",
    "run_network_simulation",
]

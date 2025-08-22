import os
import importlib.util


def find_path():
    """
    Find the directory where the cosmoGW package is installed.

    Returns
    -------
    str
        Absolute path to the cosmoGW package directory.
    """
    spec = importlib.util.find_spec("cosmoGW")
    if spec is not None and spec.submodule_search_locations:
        # This gives you the directory containing cosmoGW
        return os.path.abspath(spec.submodule_search_locations[0]) + '/'
    else:
        print('cosmoGW cannot be found, make sure you are not using an '
              'environment named cosmoGW')
        return None


# Set the COSMOGW_HOME variable to the path where cosmoGW is installed
COSMOGW_HOME = find_path()

# Explicitly import submodules so they are available as cosmoGW.<submodule>
from . import cosmology
from . import GW_analytical
from . import GW_models
from . import GW_templates
from . import hydro_bubbles
from . import interferometry
from . import plot_sets

# Optionally, define __all__ for clarity
__all__ = [
    "cosmology",
    "GW_analytical",
    "GW_models",
    "GW_templates",
    "hydro_bubbles",
    "interferometry",
    "plot_sets",
    "COSMOGW_HOME"
]

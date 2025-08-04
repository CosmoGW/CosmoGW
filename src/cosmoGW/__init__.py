def find_path():

    ## find the directory where cosmoGW is installed within sys.path
    import sys
    import os
    found = False
    paths = sys.path
    for path in paths:
      subdirs = os.walk(path)
      subdirs = list(subdirs)
      for j in subdirs:
        if not 'test' in j[0] and not 'env' in j[0]:
            if 'cosmoGW' in j[0]:
                pth = j[0]
                found = True
                break
      if found: break
    return pth + '/'

COSMOGW_HOME = find_path()

import cosmoGW.GW_back
import cosmoGW.cosmology
import cosmoGW.GW_analytical
import cosmoGW.GW_models
import cosmoGW.GW_templates
import cosmoGW.hydro_bubbles
import cosmoGW.interferometry
import cosmoGW.plot_sets

# SIMULATOR


The function of each file is listed here:

**\_\_init\_\_.py**: required to make the simulator dir a package? 

**dynamics.py**: general dynamics framework. *dynamics* fns returns an initializer and a step function that increments the system.

**render.py**: render the simulation data provided, returns an .mp4 file

**utils.py**: force and energy functions for 2-body interactions, and some data processing tools

**force.py**: force and energy functions for many-body interactions, basis representation for a general force *F*(v, d)

**environment.yml**: virtual env documentation

Required variables: Variables which define the system and the simulation params. Must be provided prior to running any code:
    box_size    : float specifying side length of box
    N           : number of particles in system
    SIMUL_STEPS : number of simulation saves performed
    dt          : simulation time step
    DELTA       : number of time steps between simulation save

Files documentation: The function of each file is listed here:
    __init__.py     : (idk what this does but apparently its required to make the simulator dir a package)
    init.py         : initializer for a given system
    particle.py     : defines the Particle namedtuple, the primary datatype used in the simulations
    dynamics.py     : general dynamics framework
    render.py       : render the simulation data provided, returns an .mp4 file
    simulate.py     : initiates simulation, returns list of states
    environment.yml : virtual env documentation

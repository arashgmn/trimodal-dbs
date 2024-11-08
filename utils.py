import brian2 as b2
from brian2.units import *

import numpy as np
from scipy.optimize import root

from configs import cfg_sim
import electrophys



def getd(D):
    """
    The relationship between the inner and outer diameter (both in um) according to
    Astrom.
    
    Args:
        D (float): The outer diameter in um.
    
    Returns:
        float: The inner diameter in um.
    """
    return D*0.74*(1-np.exp(-D/1.15))


def getL(d):
    """
    The relationship between the internodal length and the inner diameter 
    (both in um) according to Astrom.
    
    Args:
        d (float): The inner diameter in um.
    
    Returns:
        float: The internodal length in um.
    """
    return 146*d**1.12
    

def mesh(l, L, d, D, N_rep, N_in=1, N_tran=0, mode=None):
    """
    Creates a mesh for axon discretization with different segment progression types.

    Parameters
    ----------
    l : float
        Length of node of Ranvier segments in um (unitless)
    L : float
        Length of internodal sections in um (unitless)
    d : float
        Inner diameter in um (unitless)
    D : float
        Outer diameter in um (unitless)
    N_rep : int
        Number of repetitions of the node-internode pattern
    N_in : int, default=1
        Number of segments per internodal section
    N_tran : int, default=0
        Number of segments in transition regions
    mode : {'smooth', 'constant', 'equal', None}, default=None
        Meshing mode for transition regions:
        - 'smooth': Adjusts segment lengths in transition regions for gradual 
          change from l (near node) to L/N_in (near internode). Prevents abrupt 
          size changes but transition length depends on N_rep.
        - 'constant': Sets total transition region length to L/N_in while 
          maintaining l-sized segments near nodes. May cause abrupt size changes 
          near internodal segments. Falls back to equal sizes if inputs are invalid.
        - 'equal': Uses equal-length segments throughout transition region, 
          possibly different from other regions.
        - None: Disables transition regions (sets N_tran to 0). Uses one segment 
          of length l for transition coverage.

    Returns
    -------
    tuple
        - rr (int): Ranvier-to-Ranvier index difference
        - n_seg (int): Total number of segments
        - seg_loc (ndarray): Segment border locations
        - seg_diams (ndarray): Segment diameters
        - D_myelin (ndarray): Outer diameters including myelin

    Notes
    -----
    Every node of Ranvier is modeled as one segment. Internodal sections are 
    divided into N_in equal-length segments. Transition regions are divided 
    into N_tran segments with lengths determined by the selected mode.

    For adaptive meshing (modes other than None), N_tran must be >= 2.
    
    (unitless) means the provided argument should not be a Brian quantity.
    """
    
    if type(mode)==type(None):
        N_tran = 0
        
    if N_tran:
        assert N_tran>=2, "N_tran must be >= 2 for adaptive meshing!" 
        if mode=='equal':
            len_tran = [L/N_tran]*N_tran

        else:
            if mode=='constant':
                def progression_rate(x):
                    k = L/(l*N_in)
                    return x**(N_tran)- k*x + k - 1  

                r = root(progression_rate, [500], method='lm')
                r = r.x
            elif mode=='smooth':
                r = np.exp(1/(N_tran-1) *  np.log(L/l/N_in)) 
            else:
                raise NotImplementedError
                
    else:
        # these two lines harmonize all cases of modes later
        r = 1
        N_tran = 1 # anyway brian transfers d to D in one segment 
        
    len_tran = np.arange(N_tran)
    len_tran = list( l*r**(len_tran) )

    # length pattern of one node, one internode 
    seg_len = np.concatenate(    
        (
         [l],
         len_tran, 
         N_in*[L/N_in],
         len_tran[::-1],
        )
    )

    seg_len = np.tile(seg_len, N_rep) # N_rep copy of the same pattern
    seg_len = np.concatenate(([0], seg_len, [l])) # append border nodes
    seg_loc = np.cumsum(seg_len) # border locations

    # border diamters pattern
    seg_diams = d*np.ones((1+2*N_tran+N_in)*N_rep+2)

    # diameter with myelin (used lated for adjusting C and R) 
    # also useful for visualization
    D_myelin = np.concatenate(
        ([d],
        [d+(D-d)/sum(len_tran)*x for x in np.cumsum(len_tran)],
        N_in*[D], # constant
        [D-(D-d)/sum(len_tran)*x for x in np.cumsum(len_tran[::-1])],
        )
    )
    D_myelin = np.tile(D_myelin, N_rep) # N_rep copy of the same pattern
    D_myelin = np.concatenate(([d], D_myelin, [d])) # append the boundaries
    
    rr = 1+2*N_tran+N_in # index difference between two node of ranviers 
    n_seg= rr*N_rep+1 #  = morpho.n
    return rr, n_seg, seg_loc, seg_diams, D_myelin



def straight_axon(l, D, N_rep, N_in, N_tran, mode=None):
    """
    Creates a straight axon with specified morphological parameters.

    Parameters
    ----------
    l : float
        Length of node of Ranvier segments in um
    D : float
        Outer diameter in um
    N_rep : int
        Number of repetitions of the node-internode pattern
    N_in : int
        Number of segments per internodal section
    N_tran : int
        Number of segments in transition regions
    mode : {'smooth', 'constant', 'equal', None}, default=None
        Meshing mode for transition regions

    Returns
    -------
    tuple
        - morpho : brian2.Section
            Morphology object representing the axon
        - morpho_params : dict
            Dictionary containing:
            - d : Quantity
                Inner diameter
            - D : Quantity
                Outer diameter
            - D_myelin : Quantity
                Myelin diameter distribution
            - rr : int
                Ranvier-to-Ranvier index difference
            - idx_nn : array
                Indices of internodal segments
            - idx_nr : array
                Indices of Ranvier nodes
    """
    
    # morphology sizes
    d = getd(D)
    L = getL(d)
    
    if type(mode)==type(None):
        # we need to modify it to minimize the border artifacts
        N_tran = 1
        N_in = int(round(L/l))
    
    rr, n_seg, seg_loc, seg_diams, D_myelin = mesh(l, L, d, D, 
                                                   N_rep, N_in, 
                                                   N_tran, mode)
    
    # segment indices
    idxs = np.arange(n_seg, dtype=int)
    idx_nn = idxs[idxs%rr!=0] # index of INternode segments
    idx_nr = idxs[idxs%rr==0] # index of Nodes of Ranvier segments
    
    # placing segments on a stright line
    x = seg_loc-seg_loc.mean() 
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    morpho = b2.Section(n=n_seg, diameter= seg_diams*um,
                        x = x*um, y= y*um, z= z*um)
    
    morpho_params = {'d':d*um, 'D':D*um, 'D_myelin':D_myelin*um,
                    'rr': rr, 'idx_nn': idx_nn, 'idx_nr': idx_nr
                    }
                     
    return morpho, morpho_params


def get_total_length(l, D, N_in, N_tran, N_rep, mode=None):
    """
    Computes the total length of a straight axon.

    Parameters
    ----------
    l : float
        Length of node of Ranvier segments in um
    D : float
        Outer diameter in um
    N_in : int
        Number of segments per internodal section
    N_tran : int
        Number of segments in transition regions
    N_rep : int
        Number of repetitions of the node-internode pattern
    mode : {'smooth', 'constant', 'equal', None}, default=None
        Meshing mode for transition regions

    Returns
    -------
    float
        Total length of the axon in um
    """
       
    d = getd(D)
    L = getL(d)
    if type(mode)==type(None):
        # we need to modify it to minimize the border artifacts
        N_tran = 0
        N_in = int(round(L/l))
    
    rr, n_seg, seg_loc, seg_diams, D_myelin = mesh(l, L, d, D, 
                                                   N_rep, N_in, 
                                                   N_tran, mode)
    return seg_loc[-1] - seg_loc[0]
    
    
    
def curved_axon(x,y,z, l, D, N_in, N_tran=0, 
                unit=1*um, mode=None):
    """
    Makes a curved axon by placing the morphological segments on a 
    Bspline approximation of the curve. Note that all spatial sizes
    must be dimensionless quantity, yet in the same unit. i.e., m,
    mm, um, etc.
    
    Important Note:
    ===============
    For a stable Bspline fit, the distance between points must be
    roughly of the same order of magnitude.  
    
    Arguments
    =========
    x,y,z : coordinates of the desired curve 
    l     : length of node of Ranvier 
    D     : diameter of the internodal compartment 
    N_in  : number of segments per internodal compartment
    N_tran: number of segments in transition area of axon
    unit  : the dimension of sizes in brian units
    
    Returns
    =======
    morpho : The morphology of the axon
    Nrr    : Ranvier-to-Ranvier index difference (thus rr)
    nn     : INternode index (thus nn)
    nr     : Nodes of Ranvier index (thus nr)
    """
    from scipy.interpolate import make_interp_spline
    
    if type(x)==type(1*meter):
        # if inputs already have a unit, they will loose it upon stacking, but
        # so we need to multiply them by the length unit. Otherwise, (when the
        # coordinates are unitless), the unit are given by the user.
        unit = 1*meter

    fiber_coords = np.stack((x,y,z), axis=1)
    
    # morphology sizes
    d = getd(D) # always in um
    L = getL(d) # always in um
    
    if type(mode)==type(None):
        # we need to modify it to minimize the border artifacts
        N_tran = 0
        N_in = int(round(L/l))
    
    # we actually don't know N_rep so, we estimate an upper bound 
    # for it by assuming the minimum length for transition regions
    # i.e. l. 
    fiber_length = sum(np.linalg.norm(np.diff(fiber_coords, axis=0), axis=1))
    N_rep = int(round( fiber_length*unit/((l+L+2*l)*um) )) 
    
    rr, n_seg, seg_loc, seg_diams, D_myelin = mesh(l, L, d, D, 
                                                   N_rep, N_in, 
                                                   N_tran, mode)
    
    # segment indices
    idxs = np.arange(n_seg, dtype=int)
    idx_nn = idxs[idxs%rr!=0] # INternode segments
    idx_nr = idxs[idxs%rr==0] # Nodes of Ranvier segments
    
    # placing segments on the spline
    bspl = make_interp_spline(x = np.linspace(0,1,num=len(fiber_coords),endpoint=True),
                              y = fiber_coords.T, k=3, axis=1)
    
    maxiter = 5
    s = 0
    r = bspl(s)
    for idx in range(1, len(seg_loc)):
        dl = (seg_loc[idx] - seg_loc[idx-1])*um/unit
        
        # initial guess
        s_ = s
        for iter_ in range(maxiter):
            bspl_prime = bspl.derivative()(s_)
            ds = dl/np.linalg.norm(bspl_prime)
            s_ = s + ds/2 # interior point
        
        bspl_prime = bspl.derivative()(s_)
        ds = dl/np.linalg.norm(bspl_prime)
        s += ds
        r = np.append(r,bspl(s))
    
    x,y,z = r.reshape(-1,3).T
    morpho = b2.Section(n=n_seg, diameter= seg_diams*um,
                        x = x*unit, y= y*unit, z= z*unit)
    
    morpho_params = {'d':d*um, 'D':D*um, 'D_myelin': D_myelin*um, # always um
                    'rr': rr, 'idx_nn': idx_nn, 'idx_nr': idx_nr
                    }
    return morpho, morpho_params
    
    
    
def setup_neuron(morpho = None, morpho_params=None, 
                 N = None, 
                 point_current=False, 
                 method='euler', #exponential_
                 elecphys_params='Astrom',
                 name=None
                  ):
    """
    Creates and initializes a neuron or axon model with specified parameters.

    Parameters
    ----------
    morpho : Morphology, optional
        Brian2 morphology object defining the neuron geometry
    morpho_params : dict, optional
        Dictionary of morphological parameters for myelinated segments
    N : int, optional
        Number of neurons (for NeuronGroup initialization)
    point_current : bool, default=False
        If True, uses point current model
        If False, uses spatial current distribution
    method : str, default='euler'
        Integration method for differential equations
    elecphys_params : str, default='Astrom'
        Electrophysiological parameter set to use
    name : str, optional
        Name identifier for the neuron

    Returns
    -------
    Neuron
        Brian2 SpatialNeuron or NeuronGroup object with initialized states
    """
    
    namespace= electrophys.params(elecphys_params)
    I_unit = 1*amp/meter**2
    
    if type(morpho)!=type(None):
        if point_current:
            model = electrophys.eqs_spatial_point()
            I_unit = 1*amp 
        else:
            model = electrophys.eqs_spatial()
            
        neuron = b2.SpatialNeuron(morphology=morpho, model=model,
                                  namespace= namespace,
                                  Cm=namespace['Cm0'],  
                                  Ri=namespace['Ri0'], 
                                  method=method, 
                                  refractory="m > 0.75", threshold="m > 0.8",
                                  reset='',
                                  )
    else:
        assert type(N)==int
        model = electrophys.eqs_group()
        neuron = b2.NeuronGroup(N=N, model=model, namespace=namespace,
                                method=method)
    
    neuron.I = 0*I_unit
    neuron.v = namespace['El']
    neuron.h_ = 0 # this is critical since we should have no Na inactivation.
    neuron.m_ = 0
    neuron.n_ = .25
    neuron.s_ = .2 

    neuron.gKf = namespace['gKf0']
    neuron.gKs = namespace['gKs0']
    neuron.gl = namespace['gl0']
    neuron.p_Na = namespace['p_Na0']
    neuron.Cm = namespace['Cm0']
    
    # chaning the internodal parameter if morpho_params is provided
    if type(morpho)!= type(None):
        if type(morpho_params)!=type(None):
            
            # Diameter with myellination 
            neuron.add_attribute('D_myelin'); 
            neuron.D_myelin = morpho_params['D_myelin']

            # the node-to-node distance
            neuron.add_attribute('rr'); 
            neuron.rr = morpho_params['rr']

            # the index of non-nodal segemnts
            neuron.add_attribute('idx_nn'); 
            neuron.idx_nn = morpho_params['idx_nn']

            # the index of nodes of ranvers
            neuron.add_attribute('idx_nr'); 
            neuron.idx_nr = morpho_params['idx_nr']

            # lamella properties
            g_ratio = morpho_params['d']/morpho_params['D']
            n_l = (1-g_ratio)*morpho_params['D']/(2*namespace['th_lam']) 
            g_in0 = (1+g_ratio)/(2*n_l)*namespace['g_lam']
            c_in0 = (1+g_ratio)/(2*n_l)*namespace['c_lam']

            # adjust electrophysiological properties according to diameter
            D_myelin = (neuron.D_myelin[:-1] + neuron.D_myelin[1:])/2 # D at the center
            D_myelin = np.asarray(D_myelin)
            d = morpho_params['d']/(1*meter)
            D = morpho_params['D']/(1*meter)
            my_rate = (D_myelin - d)/(D-d) # myelin_ratio
    
            neuron.Cm = (1-my_rate)*namespace['Cm0'] + c_in0*my_rate
            neuron.gl = (1-my_rate)*namespace['gl0'] + g_in0*my_rate
            neuron.gKs= (1-my_rate)*namespace['gKs0'] # zero gKs on Schwann cell
            neuron.gKf= (1-my_rate)*namespace['gKf0'] # zero gKf on Schwann cell
            neuron.p_Na=(1-my_rate)*namespace['p_Na0'] # zero p_Na on Schwann cell

    return neuron
    
def compute_v_ext(r, 
          r_elec,
          I_tot,
          sigma = 0.2*siemens/meter):
    """
    Computes extracellular voltage at given locations assuming point-source current injection.

    Parameters
    ----------
    r : array-like
        Observation point coordinates
    r_elec : array-like
        Electrode position coordinates
    I_tot : Quantity
        Total injected current
    sigma : Quantity, default=0.2*siemens/meter
        Medium conductivity

    Returns
    -------
    Quantity
        Extracellular voltage at observation points
    """

    
    # loosing units
    r_ = np.asarray(r)
    r_elec_ = np.asarray(r_elec)
    d = np.linalg.norm(r_-r_elec_, axis=1)
    return I_tot/(4*np.pi*sigma*(d*1*meter))
    
    
def compute_af(neuron, r_elec, I_tot=5*mA, only_ranvier=True):
    """
    Computes the activation function along an axon based on extracellular potential gradients.

    Parameters
    ----------
    neuron : SpatialNeuron
        A Brian2 SpatialNeuron object containing morphological information
    r_elec : array-like
        Electrode position coordinates [x, y, z]
    I_tot : Quantity, default=5*mA
        Total stimulation current
    only_ranvier : bool, default=True
        If True, computes AF only at nodes of Ranvier
        If False, computes AF at all compartments

    Returns
    -------
    ndarray
        Activation function values normalized by surface area
        First and last compartments have AF=0
    """
    
    # all unitless  
    # arc length from the beginning to the seg center  
    l = neuron.morphology.distance/(1*meter) 
    
    # starting and ending diamter of each segment, and lateral area
    d_start = neuron.morphology.start_diameter/(1*meter)
    d_end = neuron.morphology.end_diameter/(1*meter)
    area = neuron.morphology.area/(1*meter**2)

    # neuron.morphology.coordinates gives the borders. We need the mid points 
    coords = np.array([neuron.morphology.x, 
                       neuron.morphology.y, 
                       neuron.morphology.z]) # loses units
    if only_ranvier:
        l = l[neuron.idx_nr]
        coords = coords[:,neuron.idx_nr]
        area = area[neuron.idx_nr]
        d_start = d_start[neuron.idx_nr]
        d_end = d_end[neuron.idx_nr]
        
    v = compute_v_ext(coords.T*1*meter, r_elec, I_tot=I_tot)/(1*volt) # unitless
    
    af = np.zeros(len(v)) # we don't compute AF for the first and last element

    for i in range(len(v)):
        A_start = np.pi*d_start[i]**2/4
        A_end = np.pi*d_end[i]**2/4
        if (i==0) or (i==len(v)-1):
            continue
        else:            
            af[i] = A_end*(v[i+1]-v[i])/(l[i+1]-l[i]) \
                  - A_start*(v[i]-v[i-1])/(l[i]-l[i-1])
    
    af /= area # to make it per unit surface
    
    return af/neuron.Ri*(1*volt/meter)
    
def setup_sweeps():
    """
    Sets up parameter ranges for simulation sweeps based on configuration file.

    Returns
    -------
    tuple
        - I_sweep : Quantity array
            Stimulation current values
        - r_sweep : Quantity array
            2D array of electrode positions
        - k_sweep : array
            Curvature parameter values

    Notes
    -----
    Parameter ranges are read from cfg_sim dictionary in the config file.
    Current sweep excludes zero and includes both positive and negative values.
    """
    
    k_sweep = np.linspace(cfg_sim['k']['start'],
                          cfg_sim['k']['end'],
                          cfg_sim['k']['N'])

    # drops 0
    I_sweep = np.linspace(cfg_sim['I']['start'],
                          cfg_sim['I']['end'],
                          cfg_sim['I']['N'])
    I_sweep = np.concatenate([I_sweep[1:], -I_sweep[1:]]) 
    I_sweep*= eval(cfg_sim['I']['unit'])
    
    # a 2D array
    r_sweep = np.zeros((cfg_sim['r']['N'],3))
    r_sweep[:,1] = np.linspace(cfg_sim['r']['start'],
                   cfg_sim['r']['end'],
                   cfg_sim['r']['N'])
    r_sweep*= eval(cfg_sim['r']['unit'])
    
    return I_sweep, r_sweep, k_sweep



def categorize(spk_count):
    """Categorizes communication mode based on the number spike 
    observed at the begeinning and the end of the axon. 
    
    If no spikes is generated on some compartment but stays localized,
    the axon is categorzied as "blocking", if only on end recives the
    spikes it is classified as "unidirectional" and otherwise
    "bidirectional". 
    
    
    Args:
        spk_count (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    n_start, n_end = spk_count[0], spk_count[-1]
    if max(spk_count) == 0:
        return 'none'

    else:
        if (n_start + n_end) > 0 :
            if n_start == n_end:
                return 'bi'
            else:
                return 'uni'
        else:
            return 'blk'
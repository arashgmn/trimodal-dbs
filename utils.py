import brian2 as b2
from brian2.units import *

import numpy as np
from scipy.optimize import root

from configs import cfg_sim
import electrophys


def getd(D):
    """
    The relationship between inner and outer diamter (both in um) according to
    Astrom.
    """
    return D*0.74*(1-np.exp(-D/1.15))


def getL(d):
    """
    The relationship between the internodal length and the inner diamter 
    (both in um) according to Astrom.
    """
    return 146*d**1.12
    

def mesh(l, L, d, D, N_rep, N_in=1, N_tran=0, mode=None):
    """
    meshes the axon with progressions of different types. All inputs must be 
    unitless (not a brian quantity) but expressed in um. The outputs, will be
    the same (in um but unitless).
    
    The length of transition regions (immediately before and 
    after a node of Ranvier) depends on the `mode`. See below.
    
    Note
    ====
    Every node of ranvier is always modeled as one segment but internodal 
    sections are divided to `N_in` segments with equal lengths. While 
    transition regions are always divided to `N_in` segments whose lengths
    (possibly) changes progressively depending on the `mode`:
    
    - `mode==None`: All segments will have equal of length of `l` (no adaptive
                    meshing). The transition region is covered with one segment
                    of length l. Instead of `mode`, one can equivalently set the
                    number of transition segments to zero. 
                    
    - `mode=='smooth'`: adjusts the length of each segments in the transition 
                        regions such that the length of segment adjacent to a
                        node of ranvier is `l` and the size of the segment 
                        adjacent to the internodal section is `L/N_in`. Thus,
                        prevents abrupt change in the descretization sizes, but
                        the length of region will depend on the `N_rep`.
    
    - `mode=='constant'`: adjusts the length of each segments in the transition
                          regions such that total lengh of a transition region 
                          is `L/N_in` (i.e., equal to the segments in internodal
                          section, although any other value can work as well), 
                          while keeping fixing the length of the segments 
                          adjacent to a node of Ranvier equal to `l` (like 
                          `smooth`). This may cause abrupt change in segment 
                          sizes near the internodal segments. For bad inputs, 
                          segments will be of equal size (and violate the 
                          assumption of constant total length). 

    - `mode=='equal`: all the transition segments will have equal lengths, but 
                      possibly different from other regions. 
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
    D is only used to estimate the for
    Returns
    =======
    morpho : The morphology of the axon
    Nrr    : Ranvier-to-Ranvier index difference (thus rr)
    nn     : INternode index (thus nn)
    nr     : Nodes of Ranvier index (thus nr)
    """
    
    # morphology sizes
    d = getd(D)
    L = getL(d)
    
    if type(mode)==type(None):
        # we need to modify it to minimize the border artifacts
        N_tran = 0
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
    computes the total length of a straight axon with the given 
    arguments (unitless but in um).
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
                 method='exponential_euler',
                  ):
    """
    A high-level function to setup a neuron or an axon.
    """
    
    namespace= electrophys.params()
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
                                  reset=''
                                 )
    else:
        assert type(N)==int
        model = electrophys.eqs_group()
        neuron = b2.NeuronGroup(N=N, model=model, namespace= namespace,
                                method=method)
    
    neuron.I = 0*I_unit
    neuron.v = namespace['El']
    neuron.h = 0 # this is critical since we should have no Na inactivation.
    neuron.m = 0
    neuron.n = .25
    neuron.s = .2 

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
          sigma = 0.3*siemens/meter):
    """
    computes the extracellular voltage on the given location(s) assuming an
    isotropic an homogeneous medium with the conductivity `sigma` under a 
    point-wise current injection. 
    """
    
    # loosing units
    r_ = np.asarray(r)
    r_elec_ = np.asarray(r_elec)
    d = np.linalg.norm(r_-r_elec_, axis=1)
    return I_tot/(4*np.pi*sigma*(d*1*meter))
    
    
def compute_af(neuron, r_elec, I_tot=5*mA, only_ranvier=True):
    """
    computes the activation function based on the location of the nodes and
    the axial reistivity. Both must be accessible through the 
    object `neuron`. On the axon ends no AF is computed.
    
    If `only_ranvier=True`, the internodal segments will be ignored and AF is
    estimated based on the location of nodes of ranvier only.
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
    Sets the parameters range for sweeping based on the values
    in the config file (cfg_sim dictionary).
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

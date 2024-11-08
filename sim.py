from utils import *
from vis import *
from configs import cfg_axon
from vis import plot

import pandas as pd

import pickle
from time import time
import sys
import os 
osjoin = os.path.join # an alias for convenience

b2.defaultclock.dt = 10*us


def run_experiment(irk=None, stim_dur=None, biphasic_ratio=10., with_ranv=True, root= '.'):
    """
    Runs a deep brain stimulation (DBS) experiment by simulating axon responses to electrical stimulation.

    The function simulates both a full axon model and optionally a nodes-of-Ranvier-only model, applying
    biphasic stimulation and analyzing signal propagation under different geometric configurations.

    Parameters
    ----------
    irk : dict, optional
        Dictionary containing stimulation parameters:
        - 'I': Stimulation current (in mA)
        - 'r': Electrode position as [x,y,z] coordinates (in mm)
        - 'k': Axon curvature parameter
        If None, uses default sweep values from config.

    stim_dur : float, optional
        Duration of the stimulation pulse in microseconds.
        If None, defaults to 70 microseconds.

    biphasic_ratio : float, default=10.
        Ratio between primary and counter pulse duration.
        Set to 0 to disable biphasic stimulation.

    with_ranv : bool, default=True
        If True, simulates both full axon and nodes-of-Ranvier-only models.
        If False, only simulates full axon model.

    root : str, default='.'
        Root directory for saving results.

    Returns
    -------
    None
        Results are saved to disk:
        - Visualization plots in results directory
        - categories.csv containing signal propagation classifications
        - errors.csv containing comparison between full and Ranvier-only models

    Notes
    -----
    The function performs:
    1. Axon geometry setup (straight and curved configurations)
    2. Neuron model initialization
    3. Simulation with warmup period
    4. Application of DBS pulses
    5. Optional physiological pulse injection
    6. Results visualization and categorization
    """

    if type(irk)==type(None):
        # setting up different cases
        I_sweep, r_sweep, k_sweep = setup_sweeps()    
    else:
        k_sweep = [irk['k']]
        r_sweep = [irk['r']] # must have unit
        I_sweep = [irk['I']] # must have unit

    if type(stim_dur)==type(None):
        stim_dur = 70*us
    
    # reading what has been done before: columns: I,r,k,runtime
    # otherwise makes a new file
    savepath = osjoin(root,'results')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        first_run = True
    # else:
    #     if os.path.exists(osjoin('.', 'results','runtimes.csv')):
    #         finished = np.loadtxt('results/runtimes.csv', ndmin=2,
    #         					  delimiter=',' , dtype=str, encoding='utf-8',
    #         					  converters = lambda x: float(x.split(' ')[0]))
    
    
    # To mimic a neurophysiological (healthy) signal, it suffices to 
    # sttongly stimulate one end of the axon. We do so by adding an
    # activation function, AF0 at either end of the axon. 
    AF0 = 500* b2.amp/(b2.meter**2)
    
    categories = []
    errors = []
    for k in np.roll(k_sweep,-1):
        start = time()
        b2.start_scope()
        
        ## AXON GENERATION
        
        # we initialize the axon as a straight one first, and then
        # modify it to a curved one
        cfg_a = cfg_axon.copy() # configuration for a straight axon
        morpho, morpho_params = straight_axon(**cfg_a) # morphology object and its parameters
        
        # let's read the x coordinates of the axon components
        x = np.array(morpho.x) # np.arrary converts to SI
        cfg_a['x'] = x
        
        # and then modify it to make a curved one. Multiple forms are possible 
        # (Uncomment others curves if you'd like to explore more...)
        cfg_a['y'] = k*(x-x.mean())**2 # parabola
        # cfg_a['y'] = x.max()*np.sin(k*(x-x.mean())*5) # sin
        # cfg_a['y'] = k*(x-x.mean())/100  #line
        # cfg_a['y'] = x.max()/(1+np.exp(k*(x-x.mean()))) #sigmoid
        #           - np.array(cfg_e['r_elec'][1]*eval(cfg_e['r_unit']))
        cfg_a['z'] = np.zeros_like(x)
        
        cfg_a.pop('N_rep') # we don't need N_rep anymore
        cfg_a['unit'] = 1*meter # because x,y,z are now in SI

        # now, this makes a new axon which is curved
        morpho, morpho_params = curved_axon(**cfg_a)
        
        # from which we can create a compartmentalized neuron and associated monitors.
        # NOTE: technically, this is still just the axon. But we follow Brian's numenclature,
        # which calls it a neuron, even though we defined no soma or dendrites...
        nrn_full = setup_neuron(morpho, morpho_params)
        mon_full = b2.StateMonitor(nrn_full, ['v','m'], record=True)
        spk_full = b2.SpikeMonitor(nrn_full)
        
        nrns = [nrn_full]
        mons = [mon_full, spk_full]
        
        ## NODES-OF-RANVIER-ONLY NEURON
        if with_ranv:
            # if it is requested to simulate axons just with compartments only at its nodes of Ranvier,
            # we need to setup a new compartmentalized neuron (axon). This one is identical to the 
            # other object. However, the activation functions later will be applied only on the Ranvier
            # nodes and the other compartments sense nothing of the extracellular potential gradient.
            nrn_ranv = setup_neuron(morpho, morpho_params)
            mon_ranv = b2.StateMonitor(nrn_ranv, ['v','m'], record=True)
            spk_ranv = b2.SpikeMonitor(nrn_ranv)
            
            nrns += [nrn_ranv]
            mons.append(mon_ranv)
            mons.append(spk_ranv)


        ## SIMULATION -- WARMUP
        # We run for a short time to relax axon to its resting state
        net = b2.Network(b2.collect())
        net.add(nrns)
        net.add(mons)
        for mon in mons:
            mon.active=False
        net.run(3*ms)  # relax the initial condition
        b2.store() # store this to be recovered later on...
        
        print('Setting up axon took ', str(round(time()-start,2)), 'seconds!') 


        ## SIMULATION
        
        # let's activate all monitors
        for mon in mons:
            mon.active=True

        # and stimulate a bunch of axons for different: 
        for id_r, r_ in enumerate(r_sweep):       # electrode position, and
            for id_I, I_ in enumerate(I_sweep):   # currents
                
                # let's make some (no too nice) names for the files
                r_str = str(np.round(r_[1].__float__(), 4)) #SI
                I_str = str(np.round(I_.__float__() * 1000 ,2)) # now in mA
                k_str = str(np.round(k,2))
                name = f'i{I_str}_r{r_str}_k{k_str}'
                
                # we need to computed activation functions for visualization
                af_full = compute_af(nrn_full, r_elec = r_, I_tot = I_, only_ranvier=False)
                af_ranv = compute_af(nrn_ranv, r_elec = r_, I_tot = I_, only_ranvier=True)
                
                
                # check if similar simulation has been carried out before
                # NOTE: It is not recommended to save the simulation results because the files are very large.
                # It's faster to run the entire simulation again. Also, the validity of test below is not well
                # tested. So use with caution.
                run_full = True
                if os.path.exists(osjoin(savepath, 'full'+name+'.pkl')):
                    print(f'{name} full already computed!')
                    run_full = False
                
                run_ranv = True
                if os.path.exists(osjoin(savepath, 'ranv'+name+'.pkl')):
                    print(f'{name} ranv already computed!')
                    run_ranv = True
                run_ranv *=  with_ranv # only run ranv if it is not computed before and also if it is requested 
                
                
                if not ((run_ranv) or (run_full)):
                    print(f'{name} already computed! Skipped.')
                else:
                    print(f'{name} started!')
                    
                    # where to place a neurophysical pulse? This pulse represent a non-pathological
                    # signal that may or may not be blocked by the DBS. None meas don't add any pulse.
                    # This pulse is always injected at the begining of the simulation...
                    for neurophys_pulse_loc in [None, 0, -1]:
                        start = time()
                        b2.restore()
		                
		                # main DBS pulse
                        if run_full:
                            nrn_full.I = af_full
                        if run_ranv:
                            nrn_ranv.I[nrn_ranv.idx_nr] = af_ranv

                        # we add healthy pulse for a brief moment only
                        if neurophys_pulse_loc is None:
                            AF0_dur = 0*second
                        else:
                            AF0_dur = 10*b2.us # very short
                            
                            if run_full:
                                nrn_full.I[neurophys_pulse_loc] = AF0
                            if run_ranv:
                                nrn_ranv.I[neurophys_pulse_loc] = AF0
                            
                            net.run(AF0_dur)

                        net.run(stim_dur-AF0_dur)
		                
		                # DBS counter pulse
                        if biphasic_ratio:
                            if run_full:
                                nrn_full.I *= -1./biphasic_ratio
                            if run_ranv:
                                nrn_ranv.I[nrn_ranv.idx_nr] *= -1/biphasic_ratio
                            net.run(stim_dur*biphasic_ratio) # counter pulse
                            
                        # A rest interval 
                        if run_full:
                            nrn_full.I *= 0
                        if run_ranv:
                            nrn_ranv.I[nrn_ranv.idx_nr] *= 0
                        net.run(10*ms) # long enough for a full spike traverse nerve

                        print(f'\tComputation finished after {round(time()-start,2)} seconds - AF0 loc: {neurophys_pulse_loc}')

                        
                        # Saving
                        # NOTE: It is not recommended to save the simulation results because the files are very large.
                        # start = time()
                        # for suffix in ['full', 'ranv']:
                        #     if eval(f'run_{suffix}'):
                        #         mon = eval('mon_'+suffix)
                        #         with open(osjoin(savepath, f'{name}_{suffix}_mon.pkl'), 'wb') as f:
                        #             pickle.dump(mon.get_states(['t', 'v']), f)
                                
                        #         spk = eval('spk_'+suffix)
                        #         with open(osjoin(savepath, f'{name}_{suffix}_spk.pkl'), 'wb') as f:
                        #             pickle.dump(spk.get_states(['count']), f)

                        # print(f'\tSaving took {round(time()-start,2)} seconds')


                        # visualization and make a summary
                        suffixs = ['full']
                        if with_ranv:
                            suffixs.append('ranv')
                            
                        for suffix in suffixs:
                            start = time()

                            # Let's plot the results for both full and ranv axons
                            nrn = eval('nrn_'+suffix)
                            if suffix == 'ranv': 
                                _af = np.zeros_like(nrn._indices())
                                _af[nrn.idx_nr] = eval('af_'+suffix)
                            else:
                                _af = eval('af_'+suffix)
                            
                            # retrieve the results: time, voltage, spike count
                            if eval('run_'+suffix):
                                t = (1*eval(f'mon_{suffix}.t')).__array__() # SI
                                v = (1*eval(f'mon_{suffix}.v')).__array__() # SI
                                spk_count = eval(f'spk_{suffix}.count')
                            else:
                                _mon = open(f'{name}_{suffix}_mon.pkl', 'rb')
                                _mon = _mon.read()
                                _mon = pickle.loads(_mon)
                                t = _mon['t'] # not SI (quantity)
                                v = _mon['v'] # not SI (quantity)
                                
                                _mon = open(f'{name}_{suffix}_spk.pkl', 'rb')
                                _mon = _mon.read()
                                spk_count = pickle.loads(_mon)
                                
                                del _mon
                                
                            if neurophys_pulse_loc == None:
                                AF0_suffix = ''
                            elif neurophys_pulse_loc == 0:
                                AF0_suffix = '_start'
                            else:
                                AF0_suffix = '_end'
                            
                            # t = mon.t[cond] * 1000 # now
                            # v = mon.v[:,cond]/volt
                            
                            # let's remove the initial relaxation period from time
                            t -= t[0]                            # SI
                            
                            # and also let's only look at the changes int the voltage w.r.t the resting potential
                            v -= nrn.namespace['El'].__array__() # SI
                            
                            cond = t<=1.6*1e-3 # we just plot until this time, becuase it's nicer visually. nothing happens after that.
                            plot(t[cond], v[:,cond], r_, _af, nrn, spk_count, 
                                    stim_dur.__float__(), biphasic_ratio, 
                                    name= name, suffix=suffix, AF0_suffix=AF0_suffix, 
                                    save_root=savepath)
                            
                            category = categorize(spk_count) # let's categorize communication mode based on the number of spikes passed.
                            categories.append([name, suffix, AF0_suffix.split('_')[-1], r_str, I_str, k_str, category]) # and save it for later
                            
                            print(f'\tViz of {suffix} took {round(time()-start,2)} seconds.')
                            

                        if with_ranv:
                            v_full = (1*mon_full.v[nrn.idx_nr,:]).__array__() # SI
                            v_ranv = (1*mon_ranv.v[nrn.idx_nr,:]).__array__() # SI
                            errors.append([name, AF0_suffix.split('_')[-1], r_str, I_str, k_str, np.max(abs(v_ranv-v_full))])

    # Summarize the classes
    df = pd.DataFrame(data=categories, columns=['name', 'suffix', 'af0', 'r', 'I', 'k', 'c'])
    if os.path.exists(osjoin(savepath,'categories.csv')):
        df_prev = pd.read_csv(osjoin(savepath,'categories.csv'))
        df = pd.concat((df_prev, df))
    df.to_csv(osjoin(savepath,'categories.csv'), index=False) 

    # and the differences between the continuum and nodes of ranvier-only
    df = pd.DataFrame(data=errors, columns=['name', 'af0', 'r', 'I', 'k', 'mae'])
    if os.path.exists(osjoin(savepath,'errors.csv')):
        df_prev = pd.read_csv(osjoin(savepath,'errors.csv'))
        df = pd.concat((df_prev, df))
    df.to_csv(osjoin(savepath,'errors.csv'), index=False) 
    
	

if __name__=='__main__':
    
    # stimulation current, I, electrode radius, r, and the curvature of axon, k, can be passed as
    # a dictionary of list with keys I, r, k
    irk = None            # none retrieves the default values from the config file
    stim_dur = None       # stimualtion duration. None retrieves the default values from the config file
    biphasic_ratio = 10   # The ratio between primary and secondary DBS pulse. Set zero to deactivate
    if len(sys.argv)>1:
        irk = {'I': float(sys.argv[1])*mA,
               'r': np.array([0, float(sys.argv[2]), 0])*mm,
               'k': float(sys.argv[3])}
        if len(sys.argv)>4:
            stim_dur = float(sys.argv[4])*us
            biphasic_ratio = float(eval(sys.argv[5]))
            
    run_experiment(irk = irk, stim_dur = stim_dur, biphasic_ratio=biphasic_ratio)

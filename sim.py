from utils import *
from vis import *
from configs import cfg_axon
from vis import plot


from time import time
import sys
import os 
osjoin = os.path.join # an alias for convenience

b2.defaultclock.dt = 5*us


def run_experiment(irk=None, stim_dur=None, biphasic_ratio=10., 
                   with_ranv=False):
    
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
    if not os.path.exists(osjoin('.', 'results')):
        os.makedirs(osjoin('.', 'results'))
        first_run = True
    else:
        if os.path.exists(osjoin('.', 'results','runtimes.csv')):
            finished = np.loadtxt('results/runtimes.csv', ndmin=2,
            					  delimiter=',' , dtype=str, encoding='utf-8',
            					  converters = lambda x: float(x.split(' ')[0]))
            
    for k in np.roll(k_sweep,-1):
        start = time()
        b2.start_scope()
        
        cfg_a = cfg_axon.copy()
        morpho, morpho_params = straight_axon(**cfg_a)

        x = np.array(morpho.x) # this brings x to SI
        cfg_a['x'] = x
        cfg_a['y'] = k*(x-x.mean())**2 #\
        #           - np.array(cfg_e['r_elec'][1]*eval(cfg_e['r_unit']))
        cfg_a['z'] = np.zeros_like(x)
        cfg_a.pop('N_rep') # we don't need N_rep anymore
        cfg_a['unit'] = 1*meter # because x,y,z are now in SI

        morpho, morpho_params = curved_axon(**cfg_a)
        
        nrn_full = setup_neuron(morpho, morpho_params)
        mon_full = b2.StateMonitor(nrn_full, ['v','m'], record=True)
        spk_full = b2.SpikeMonitor(nrn_full)
        mons = [mon_full]
        
        if with_ranv:
            nrn_ranv = setup_neuron(morpho, morpho_params)
            mon_ranv = b2.StateMonitor(nrn_ranv, ['v','m'], record=True)
            spk_ranv = b2.SpikeMonitor(nrn_ranv)
            mons.append(mon_ranv)

        # Let's start all from a relaxed state 
        net = b2.Network(b2.collect())
        net.add(mons)
        for mon in mons:
            mon.active=False
        b2.run(3*ms)  # relax the initial condition
        b2.store()

        print('Setting up axon took ', str(round(time()-start,2)), 'seconds!') 

        for mon in mons:
            mon.active=True

        for id_r, r_ in enumerate(r_sweep):
            for id_I, I_ in enumerate(I_sweep):
                
                r_str = str(np.round(r_[1],4))
                I_str = str(np.round(I_ * 1000 ,2)) # now in mA
                k_str = str(np.round(k,2))
                
                # except it is just one configuration
                if type(irk)!=None or first_run: 
                    skip=False
                
                else:
                    # we can skip if this is done before
                    cond_I = finished[:,0] == np.round(I_/(1*mA),2)
                    cond_r = finished[:,1] == np.round(np.asarray(r_)[1],4) # has no units
                    cond_k = finished[:,2] == np.round(k,2)
                    skip = sum(cond_r * cond_I * cond_k)
                
                if not skip:
                    print('i=', I_str,'r=', r_str, 'k=', k_str, 'started!')
                    
                    start = time()
                    b2.restore()
                    AF_full = compute_af(nrn_full, only_ranvier=False,
                                         r_elec = r_, I_tot = I_)
                    
                    # main pulse
                    nrn_full.I = AF_full
                    if with_ranv:
                        AF_ranv = compute_af(nrn_ranv, r_elec = r_, I_tot = I_)
                        nrn_ranv.I[nrn_ranv.idx_nr] = AF_ranv
                    b2.run(stim_dur) 
                    
                    # counter pulse
                    if biphasic_ratio:
                        nrn_full.I = -AF_full/biphasic_ratio
                        if with_ranv:
                            nrn_ranv.I[nrn_ranv.idx_nr] = -AF_ranv/biphasic_ratio
                        b2.run(stim_dur*biphasic_ratio) # counter pulse
                        
                    # rest
                    nrn_full.I = 0*AF_full
                    if with_ranv:
                        nrn_ranv.I[nrn_ranv.idx_nr] = 0*AF_ranv
                    b2.run(10*ms) # long enough for a full spike traverse nerve
                    
                    
                    # visualization
                    suffixs = ['full']
                    if with_ranv:
                        suffixs.append('ranv')
                    
                    for suffix in suffixs:
                        nrn = eval('nrn_'+suffix)
                        mon = eval('mon_'+suffix)
                        spk = eval('spk_'+suffix)
                        if suffix == 'ranv': 
                            af = np.zeros_like(nrn._indices())
                            af[nrn.idx_nr] = eval('AF_'+suffix)
                        else:
                            af = eval('AF_'+suffix)

                        cond = (mon.t>=2.9*ms) * (mon.t<=4.6*ms) 
                        t = mon.t[cond]/ms
                        v = mon.v[:,cond]/volt
                        name = osjoin('.', 'results',suffix+f'_i{I_str}_r{r_str}_k{k_str}')
                        
                        print(f'\tComputation finished after {round(time()-start,2)} seconds')
                        plot(t, v, r_, af, nrn, mon, spk, stim_dur, biphasic_ratio, suffix, name)
                        print(f'\tViz finished after {round(time()-start,2)} seconds')
                        
                        del mon, v, t, spk, nrn
                    
                    runtime = round(time()-start,2)
                    if type(irk)==type(None): # only write for grid search
                        with open(osjoin('.', 'results', 'runtimes.csv') ,'a') as file:
                            file.write(','.join([I_str, r_str, k_str, str(runtime)])+'\n')
                    
                    # print('i=', I_str,'r=', r_str, 'k=', k_str, 'finished after',\
                    #       runtime, 'seconds')
                else:
                    print('i=', I_str,'r=', r_str, 'k=', k_str, 'skipped!')
                        

if __name__=='__main__':
    irk = None
    stim_dur = None
    biphasic_ratio = 10 # set zero to deactivate
    if len(sys.argv)>1:
        irk = {'I': float(sys.argv[1])*mA,
               'r': np.array([0, float(sys.argv[2]), 0])*mm,
               'k': float(sys.argv[3])}
        if len(sys.argv)>4:
            stim_dur = float(sys.argv[4])*us
            biphasic_ratio = float(eval(sys.argv[5]))
            
    run_experiment(irk = irk, stim_dur = stim_dur, biphasic_ratio=biphasic_ratio)

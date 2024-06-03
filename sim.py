from utils import *
from vis import *
from configs import cfg_axon
from vis import plot

import pickle
from time import time
import sys
import os 
osjoin = os.path.join # an alias for convenience

b2.defaultclock.dt = 10*us


def run_experiment(irk=None, stim_dur=None, biphasic_ratio=10., 
                   with_ranv=True, root= '.'):
    
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
        
        nrns = [nrn_full]
        mons = [mon_full, spk_full]
        if with_ranv:
            nrn_ranv = setup_neuron(morpho, morpho_params)
            mon_ranv = b2.StateMonitor(nrn_ranv, ['v','m'], record=True)
            spk_ranv = b2.SpikeMonitor(nrn_ranv)
            
            nrns = [nrn_ranv]
            mons.append(mon_ranv)
            mons.append(spk_ranv)

        # Let's start all from a relaxed state 
        net = b2.Network(b2.collect())
        net.add(nrns)
        net.add(mons)
        for mon in mons:
            mon.active=False
        net.run(3*ms)  # relax the initial condition
        b2.store()

        print('Setting up axon took ', str(round(time()-start,2)), 'seconds!') 

        for mon in mons:
            mon.active=True

        for id_r, r_ in enumerate(r_sweep):
            for id_I, I_ in enumerate(I_sweep):
                r_str = str(np.round(r_[1].__float__(), 4)) #SI
                I_str = str(np.round(I_.__float__() * 1000 ,2)) # now in mA
                k_str = str(np.round(k,2))
                name = f'i{I_str}_r{r_str}_k{k_str}'
                
                # name = osjoin('.', 'results', )
                        
                # except it is just one configuration
                # if type(irk)!=None or first_run: 
                #     skip=False
                
                run_full = True
                if os.path.exists(osjoin(savepath, 'full'+name+'.pkl')):
                    print(f'{name} full already computed!')
                    run_full = False
                
                run_ranv = True
                if os.path.exists(osjoin(savepath, 'ranv'+name+'.pkl')):
                    print(f'{name} ranv already computed!')
                    run_ranv = True
                run_ranv *=  with_ranv # only run ranv if it is not computed before and also if it is requested 
                
                    # else:
                    #     skip = False
                    # # we can skip if this is done before
                    # cond_I = finished[:,0] == np.round(I_/(1*mA),2)
                    # cond_r = finished[:,1] == np.round(np.asarray(r_)[1],4) # has no units
                    # cond_k = finished[:,2] == np.round(k,2)
                    # skip = sum(cond_r * cond_I * cond_k)
                
                # we need to computed them for visualization
                af_full = compute_af(nrn_full, r_elec = r_, I_tot = I_, only_ranvier=False)
                af_ranv = compute_af(nrn_ranv, r_elec = r_, I_tot = I_, only_ranvier=True)
                
                if (run_ranv) or (run_full):
                    print(f'{name} started!')
                    
                    start = time()
                    b2.restore()
                    
                    # Stimulation
                    
                    # main pulse
                    if run_full:
                        nrn_full.I = af_full
                    if run_ranv:
                        nrn_ranv.I[nrn_ranv.idx_nr] = af_ranv
                    net.run(stim_dur) 
                    
                    # counter pulse
                    if biphasic_ratio:
                        if run_full:
                            nrn_full.I *= -1./biphasic_ratio
                        if run_ranv:
                            nrn_ranv.I[nrn_ranv.idx_nr] *= -1/biphasic_ratio
                        net.run(stim_dur*biphasic_ratio) # counter pulse
                        
                    # rest
                    if run_full:
                        nrn_full.I *= 0
                    if run_ranv:
                        nrn_ranv.I[nrn_ranv.idx_nr] *= 0
                    net.run(10*ms) # long enough for a full spike traverse nerve
                    
                    print(f'\tComputation finished after {round(time()-start,2)} seconds')
                    
                    # Saving
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
                    
                else:
                    print(f'{name} already computed! Skipped.')
                
                
                # visualization
                suffixs = ['full']
                if with_ranv:
                    suffixs.append('ranv')
                    
                for suffix in suffixs:
                    start = time()
                
                    nrn = eval('nrn_'+suffix)
                    if suffix == 'ranv': 
                        _af = np.zeros_like(nrn._indices())
                        _af[nrn.idx_nr] = eval('af_'+suffix)
                    else:
                        _af = eval('af_'+suffix)
                    
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
                        
                    # t = mon.t[cond] * 1000 # now
                    # v = mon.v[:,cond]/volt
                    t -= t[0]                            # SI
                    v -= nrn.namespace['El'].__array__() # SI
                    # cond = (t>=2.9*1e-3) * (t<=4.6*1e-3) 
                    cond = t<=1.6*1e-3
                    plot(t[cond], v[:,cond], r_, _af, nrn, spk_count, 
                         stim_dur.__float__(), biphasic_ratio, 
                         suffix, name, save_root=savepath)
                    
                    del nrn, _af, v, t, spk_count
                    print(f'\tViz of {suffix} took {round(time()-start,2)} seconds.')
                    # runtime = round(time()-start,2)
                    
                    # if type(irk)==type(None): # only write for grid search
                    #     with open(osjoin('.', 'results', 'runtimes.csv') ,'a') as file:
                    #         file.write(','.join([I_str, r_str, k_str, str(runtime)])+'\n')
                    
                    # print('i=', I_str,'r=', r_str, 'k=', k_str, 'finished after',\
                    #       runtime, 'seconds')
                        

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

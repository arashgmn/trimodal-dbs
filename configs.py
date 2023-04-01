import numpy as np

cfg_axon = dict(
    l = 1,
    D = 5.7,
    N_rep = 20,
    N_tran = 10,
    N_in = 20,
    mode=None
)

cfg_elec = dict(
    r_elec = np.array([0,2000,0]),
    I_tot = -10,
    I_unit= 'mA',
    r_unit= 'um',
)

# change this dictionary for altering the sweep range
cfg_sim = dict(
    I = {'N': 2, 'start':0, 'end': 8, 'unit':'mA'},
    r = {'N': 5 , 'start':1, 'end': 3, 'unit':'mm'},
    k = {'N': 1, 'start':0, 'end': -100, 'unit':'1'}
)

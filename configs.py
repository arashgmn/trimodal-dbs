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
   I = {'N': 8, 'start':1.2, 'end': 9.6, 'unit':'mA'}, # would be doubled to spann negatives as well
   r = {'N': 3 , 'start':1, 'end': 3, 'unit':'mm'},
   k = {'N': 11, 'start':-100, 'end': 100, 'unit':'1'}
)

# config for quick test
# cfg_sim = dict(
#     I = {'N': 2, 'start':1.2, 'end': 9.6, 'unit':'mA'}, # would be doubled to spann negatives as well
#     r = {'N': 2 , 'start':1, 'end': 3, 'unit':'mm'},
#     k = {'N': 10, 'start':-100, 'end': 100, 'unit':'1'}
# )

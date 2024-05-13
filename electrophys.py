from brian2.units import *
from brian2.units.constants import faraday_constant as F
from brian2.units.constants import gas_constant as R

def eqs_spatial():
    model='''
# The same equations for the whole neuron, but possibly different parameter values
# distributed transmembrane current
i_ion = -i_L - i_Ks -i_Kf -i_Na : amp/meter**2

i_L = gl * (v-El) : amp/meter**2

i_Ks = gKs * s * (v-Ek) : amp/meter**2
#ds/dt= (alpha_s*(1-s) - beta_s * s) * (int(s>=0) * int(s<=1)) + 5*Hz * (1-s)*(int(s<0) + int(s>1)): 1
ds/dt= alpha_s*(1-s) - beta_s * s : 1
alpha_s = Qs*0.00122*23.6/exprel(-(v+12.5*mV)/(23.6*mV))/ms : Hz
beta_s = Qs*0.000739*21.8/exprel((v+80.1*mV)/(21.8*mV))/ms : Hz

i_Kf = gKf * n**4 * (v-Ek): amp/meter**2
#dn/dt = (alpha_n * (1-n) - beta_n * n) * (int(n>=0) * int(n<=1)) + 5*Hz * (1-n)*(int(n<0) + int(n>1)) : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
alpha_n = Qn*0.00798*1.1/exprel(-(v+93.2*mV)/(1.1*mV))/ms : Hz
beta_n = Qn*0.0142*10.5/exprel((v+76.*mV)/(10.5*mV))/ms : Hz

i_Na = m**3 * h * p_Na * F*(F*v)/(R*T) * fraction: amp/meter**2
fraction = (Na_in - Na_out *exp(-(F*v)/(R*T)))/(1-exp(-(F*v)/(R*T))): mole/metre**3
#dm/dt = (alpha_m * (1-m) - beta_m * m) * (int(m>=0) * int(m<=1)) + 5*Hz * (1-m)*(int(m<0) + int(m>1)) : 1
#dh/dt = (alpha_h * (1-h) - beta_h * h) * (int(h>=0) * int(h<=1)) + 5*Hz * (1-h)*(int(h<0) + int(h>1)) : 1
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m = Qm*1.86*10.3/exprel(-(v+18.4*mV)/(10.3*mV))/ms : Hz
beta_m = Qm*0.086*9.16/exprel((v+22.7*mV)/(9.16*mV))/ms  : Hz
alpha_h = Qh*0.0336*11.0/exprel((v+111.*mV)/(11.*mV))/ms : Hz
beta_h = Qh*2.3/(1+exp(-(v+28.8*mV)/(13.4*mV)))/ms : Hz

gl: siemens/meter**2 
gKs : siemens/meter**2 
gKf : siemens/meter**2 
p_Na: meter/second 

Im = i_ion + I : amp/meter**2
I : amp/meter**2 
'''
    return model
    
def eqs_spatial_point():
    model = eqs_spatial()
    model = ''.join(model.split('Im =')[0])
    model +='''
Im = i_ion : amp/meter**2
I: amp (point current)
'''
    return model

def eqs_group():
    model= eqs_spatial()
    model+='''
dv/dt = Im/Cm : volt
Cm: farad/meter**2 
'''
    return model
    
def params(ref='Astrom'):
    
    namespace = dict(
        # lamella thickness
        th_lam = 24*nmetre,
        
        # shared parameters between Astrom and Schwartz
        El = -84*mV,
        Ek = -84*mV,
        Na_out = 154*mmole/liter,
        Ri0 = 0.4*metre/siemens, # the axial resistivity
        T0 = (20+273)*kelvin,
        g_lam = 5 *siemens/(metre**2),
        c_lam = 500 *uF/(metre**2),
        
        # Astrom parameters
        Na_in = 20*mmole/liter,  # probably wrong (doesn't match Schwartz)
        gl0 = 400*siemens/meter**2, 
        gKf0 = 300*siemens/meter**2,
        gKs0 = 600*siemens/meter**2,
        p_Na0 = 7.04e-5 *meter/second,
        Cm0 = 0.028*farad/(1*meter**2),
        # T = (37+273)*kelvin # experiment temperature
        T = (20+273)*kelvin # experiment temperature
    )
    
    if ref=='Schwartz':
        area = 50 * um**2 # Following Astrom, according to Wesselink (rough area of neuron)
        namespace['Na_in'] = 35*mmole/liter # fig 6D uses 90*mmole/liter
        namespace['gl0']   = 30e-9*siemens/area # fig 6D uses 40e-9*siemens/area (default is 30e-9)
        namespace['gKf0'] = 15e-9*siemens/area
        namespace['gKs0'] = 30e-9*siemens/area
        namespace['p_Na0'] = 3.52e-9 *cm**3/second/area
        namespace['Cm0'] = 1.4*pF/area
        namespace['T'] = (20+273)*kelvin # experiment temperature
        
    # Q10 factors
    namespace['Qm'] = 2.2**((namespace['T']-namespace['T0'])/(10*kelvin))
    namespace['Qh'] = 2.9**((namespace['T']-namespace['T0'])/(10*kelvin))
    namespace['Qn'] = 3.0**((namespace['T']-namespace['T0'])/(10*kelvin))
    namespace['Qs'] = 3.0**((namespace['T']-namespace['T0'])/(10*kelvin))
    
    # physical constants
    namespace['R'] = R
    namespace['F'] = F
    return namespace
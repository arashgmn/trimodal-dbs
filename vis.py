from brian2.units import meter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

from utils import *
from configs import cfg_axon
from pdb import set_trace

def plot_axon_native(neuron, r_elec=None, color='g', ax = None, 
                     plot_centers=True, plot_borders=True):
    """
    plots the axon and electrode location in x-y plane.
    """
    #centers (unitless)
    center_x = neuron.morphology.x_ 
    center_y = neuron.morphology.y_ 
    center_z = neuron.morphology.z_ 
    
    #borders (unitless)
    x,y,z = neuron.morphology.coordinates_.T
    
    # border diamters
    diam = np.asarray(neuron.D_myelin)
    
    if type(ax)==type(None):
        fig, ax = plt.subplots(figsize=(16,8))
    
    ax.fill_between(x,y-diam/2, y+diam/2, color=color, alpha=0.5)
    if plot_centers:
        ax.scatter(center_x, center_y, marker='x', c='r', s=20) 
    if plot_borders:
        ax.vlines(x, y-diam/2, y+diam/2, linestyle='--', color='k', linewidth=1) 
    
    ax.set_ylabel('Diamter [m]')
    ax.set_xlabel('location [m]')
    
    if type(r_elec)!=type(None):
        r_elec/=(1*meter)
        ax.scatter(r_elec[0],#/(unit),
                   r_elec[1],#/(unit), 
                   marker='x');
    return ax         
    

class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, 
                 loc=4, pad=0.1, borderpad=0.1, sep=2, 
                 prop=None, barcolor="black", barwidth=None, 
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

        
def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

    return sb   

def plot(t, v, af, nrn, mon, spk, stim_dur_ms, biphasic, suffix, name):
    fig, axs = plt.subplot_mosaic([['.', 'cb', '.'], ['af', 'sol','spk']],
                    figsize=(10,5), 
                    gridspec_kw={'width_ratios':[0.3, 1.0,.3],
                                'height_ratios':[0.05,1]},
                    )
    
    # plots
    cb = axs['sol'].imshow(v, aspect='auto', origin='lower', 
                            vmin=-0.4, vmax=0.4)
    fig.colorbar(cb, cax = axs['cb'], orientation='horizontal',)
    axs['spk'].plot(spk.count, nrn._indices())
    axs['af'].plot(af, nrn._indices(),label='primamry', color='purple')
    
    # time bar indicators
    idx_start = int(round((3-2.9)/0.01))+1
    idx_counter = int(round((3+stim_dur_ms-2.9)/0.01))+1
    idx_end = int(round((3+11*stim_dur_ms-2.9)/0.01))+1
    
    axs['sol'].plot([idx_start, idx_counter], [1000, 1000], 
                    color='purple', linewidth=5)
    # biphasic plots
    if biphasic:
        axs['af'].plot(-af/10., nrn._indices(), 
                        label='counter', color='orange')
        axs['sol'].plot([idx_counter, idx_end], [1000, 1000],
                        color='orange', linewidth=5)

    # y-axis
    for id in ['af','spk']:
        axs[id].set_ylim(nrn._indices()[0], nrn._indices()[-1])
    for id in ['sol','spk']:
        axs[id].set_yticklabels([])
    axs['af'].set_ylabel('segment index')

    # x-axis
    if suffix=='full':
        axs['af'].set_xlim(-5, 5)
    else:
        axs['af'].set_xlim(-500,500)

    axs['spk'].set_xlim(-0.2, 3.2)
    axs['spk'].set_xticks([0,1,2,3])
    axs['spk'].set_xticklabels(['0','1','2','3'])

    axs['cb'].xaxis.tick_top()

    ticks_times = t[::int(.1*ms//b2.defaultclock.dt)]
    ticks_idx = [i for i,_ in enumerate(t) if t[i] in ticks_times]
    ticks_labels = [str(round(l,3)) for l in ticks_times] 
    axs['sol'].set_xticks(ticks_idx)
    axs['sol'].set_xticklabels(ticks_labels)
    axs['sol'].set_xlabel('Time [ms]')

    # titles
    axs['spk'].set_xlabel('Spike count [unitless]')
    axs['af'].set_xlabel(r'Activation function [$A/m^2$]')
    axs['cb'].set_title('Membrane potential [V]')

    # legend
    axs['af'].legend(loc=3)
    plt.tight_layout()
    plt.savefig(name+'.png', dpi=200,
                bbox_inches='tight')
    plt.close()

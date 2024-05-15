from brian2.units import meter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
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

def plot(t, v, r, af, nrn, mon, spk, stim_dur, biphasic_ratio, suffix, name,
         amp_scale=1300, amp_axis_loc=1500):
    
    dt = mon.t[1] - mon.t[0]
    idx_start = 0# int(round((3-2.9)/0.01))+1
    idx_counter = int(round(stim_dur/dt))
    idx_end = int(round(stim_dur* (1+biphasic_ratio)/dt))

    fig, axs = plt.subplot_mosaic([['geo', 'cb', '.'], ['af', 'sol','spk']],
                    figsize=(9,5), 
                    gridspec_kw={'width_ratios':[0.25, 1.0,.1],
                                'height_ratios':[0.05,1]},
                    )
    
    # geo
    axs['geo'].plot(*r.__array__()[:2], marker='o', color='k', ms=4)
    axs['geo'].plot(nrn.morphology.x.__array__(), 
                    nrn.morphology.y.__array__(), 
                    lw=5, color='gray', alpha=.5)

    axs['geo'].scatter(nrn.morphology.x.__array__()[nrn.idx_nr], 
                    nrn.morphology.y.__array__()[nrn.idx_nr], 
                    zorder=99, marker='s', s=2,
                    color=plt.cm.Spectral(np.linspace(0,1,len(nrn.idx_nr)))
                    )
    axs['geo'].set_aspect('equal')
    axs['geo'].set_axis_off()
    axs['geo'].margins(x=.05, y=.05)
    # axs['geo'].set_xlim(axs['geo'].get_xlim(), adjustable='box', expand=True)
    # axs['geo'].set_ylim(axs['geo'].get_ylim(), adjustable='box', expand=True)
    # xlims = axs['geo'].get_xlim()
    # ylims = axs['geo'].get_ylim()
    # axs['geo'].set_xlim(xlims[0]*1.1, xlims[1]*1.1)
    # axs['geo'].set_xlim(ylims[0]*1.1, ylims[1]*1.1)
    
    
    # sol
    sol = axs['sol'].imshow(v.__array__(), 
                            vmin=-0.3-.083, vmax=0.3-.083, cmap='bwr', 
                            aspect='auto', origin='lower', )

    # cb
    fig.colorbar(sol, cax = axs['cb'], orientation='horizontal', extend='both')
    # cb = axs['sol'].imshow(v, aspect='auto', origin='lower', 
    #                         vmin=-0.4, vmax=0.4)
    
    
    # time bar indicators
    # idx_start = int(round((3-2.9)/0.01))+1
    # idx_counter = int(round((3+stim_dur_ms-2.9)/0.01))+1
    # idx_end = int(round((3+11*stim_dur_ms-2.9)/0.01))+1
    
    # psuedo axis for wavefrom:
    # we scale the amplitdue of the pulse such that it maxes at
    # 10. The amp=0 axis is fixed at y=1000
    sign = np.sign(float(name.split('_i')[1].split(' A')[0]))
    val = np.abs(float(name.split('_i')[1].split(' A')[0]))

    # axis
    axs['sol'].plot([idx_start, idx_end*1.15], [amp_axis_loc, amp_axis_loc], color='k', linewidth=0.5) # pseudo x-axis
    # axs['sol'].plot([idx_start, idx_start],[amp_axis_loc, amp_axis_loc+amp_scale*1.5], color='k', linewidth=0.5)

    # scale bar
    axs['sol'].plot([idx_end*1.2, idx_end*1.2], [amp_axis_loc, amp_axis_loc + amp_scale], color='k', linewidth=3) # scale bar
    axs['sol'].text(idx_end*1.22, amp_axis_loc, f'{val} mA', color='k', rotation=90, fontsize=8, transform=axs['sol'].transData) # scale bar text

    # waveform
    axs['sol'].plot([idx_start, idx_start, idx_counter, idx_counter],
                [amp_axis_loc, 
                 amp_axis_loc+amp_scale*sign, 
                 amp_axis_loc+amp_scale*sign,
                 amp_axis_loc], 
                color='k', linewidth=1)
    
    # pulse indicators
    axs['sol'].plot([idx_start, idx_counter],[amp_axis_loc, amp_axis_loc], 
                color='green', linewidth=2, alpha=0.5)
    
    # # axs['sol'].plot([idx_start, idx_end*1.15], [amp_axis_loc, amp_axis_loc], color='k', linewidth=0.5)
    # # axs['sol'].plot([idx_start, idx_start],[amp_axis_loc, amp_axis_loc+amp_scale*1.5], color='k', linewidth=0.5)
    
    # axs['sol'].plot([idx_start*.8, idx_start*.8], 
    #                 [amp_axis_loc, amp_axis_loc + amp_scale], 
    #                 color='k', linewidth=3)
    # # axs['sol'].text(0.6, 0.25, f'3-{val}', color='k', fontsize=12, transform=axs['sol'].transAxes)
    # # axs['sol'].text(0.5, 0.15, f'4-{val}', color='k', fontsize=12, transform=axs['sol'].transAxes)
    # axs['sol'].text(0.03, 0.13, f'{val}', color='k', rotation=90,
    #                 fontsize=8, transform=axs['sol'].transAxes)
    
    
    # axs['sol'].text(0.04, 0.25, r'$I_{elec}\ [A]$', color='k', 
    #                 fontsize=10, fontfamily='sans', 
    #                 transform=axs['sol'].transAxes)
    
    # # axs['sol'].text(0.02, 0.2*(), f'{val}', color='k', 
    # #                 fontsize=8, fontfamily='sans', 
    # #                 transform=axs['sol'].transAxes)
        
    # axs['sol'].plot([idx_start, idx_start, idx_counter, idx_counter],
    #                 [amp_axis_loc, 
    #                  amp_axis_loc+amp_scale*sign, 
    #                  amp_axis_loc+amp_scale*sign,
    #                  amp_axis_loc], 
    #                 color='k', linewidth=1)
    # axs['sol'].plot([idx_start, idx_counter],[amp_axis_loc, amp_axis_loc], 
    #                 color='green', linewidth=2, alpha=0.5)
    
   
    
    # af
    axs['af'].vlines(0, 0, len(nrn._indices()), ls=':', color='k', lw=1)
    axs['af'].plot(af, nrn._indices(), label='primary', color='green')

    # spikes
    axs['spk'].plot(spk.count, nrn._indices())
    
    # biphasic plots
    if biphasic_ratio:
        # rest of waveform
        axs['sol'].plot([idx_counter, idx_counter, idx_end, idx_end],
                    [amp_axis_loc, 
                    amp_axis_loc-amp_scale*sign/biphasic_ratio, 
                    amp_axis_loc-amp_scale*sign/biphasic_ratio,
                    amp_axis_loc], 
                    color='k', linewidth=1)
    
        # pulse indicator
        axs['sol'].plot([idx_counter, idx_end],[amp_axis_loc, amp_axis_loc], 
                        color='purple', linewidth=2, alpha=0.5 )
        
        # counter pulse
        axs['af'].plot(-af/biphasic_ratio, nrn._indices(), 
                            label='counter', color='purple')
        
        
    
    # y-axis
    for id in ['af','spk', 'sol']:
        axs[id].set_ylim(nrn._indices()[0], nrn._indices()[-1])
        axs[id].set_yticks(nrn.idx_nr, [])

    axs['af'].set_yticks(nrn.idx_nr, [u'\u25A0' for _ in nrn.idx_nr], 
                         fontname='STIXGeneral', fontsize=5)
    [label.set_color(i) for (i, label) in 
     zip(plt.cm.Spectral(np.linspace(0,1,len(nrn.idx_nr))), 
        axs['af'].yaxis.get_ticklabels())
     ]
    
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

    # ticks_times = t[::int(.1*ms//b2.defaultclock.dt)]
    # ticks_idx = [i for i,_ in enumerate(t) if t[i] in ticks_times]
    # ticks_labels = [str(round(l,3)) for l in ticks_times] 
    # axs['sol'].set_xticks(ticks_idx)
    # axs['sol'].set_xticklabels(ticks_labels)
    
    tidx = np.linspace(0, len(t) - 1, 17, dtype=int)
    axs['sol'].set_xticks(tidx, np.round(t.__array__(), 2)[tidx])
    # axs['sol'].set_xticklabels(ticks_labels)
    axs['sol'].set_xlabel('Time [ms]')

    # titles
    axs['spk'].set_xlabel('Spike count')
    axs['af'].set_xlabel(r'Activation function [$A/m^2$]')
    axs['cb'].set_title('Membrane potential [V]')

    # legend
    axs['af'].legend(loc=3)
    plt.tight_layout()
    plt.savefig(name+'.png', dpi=200,
                bbox_inches='tight')
    plt.close()

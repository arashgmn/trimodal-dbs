from brian2.units import meter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

from utils import *
from configs import cfg_axon
from pdb import set_trace

def polish(fig, axs, lw=1, tight=True):
    """
    Polishes matplotlib figure by aligning labels and simplifying axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to polish
    axs : array-like
        Axes to polish
    lw : float, default=1
        Line width for axis spines
    tight : bool, default=True
        Whether to apply tight_layout
    """
    fig.align_xlabels(axs)
    fig.align_ylabels(axs)

    axs = np.array(axs)
    for ax in axs.ravel():
        simpleaxis(ax, lw)

    if tight:
        plt.tight_layout()
        
        
        
def simpleaxis(ax, lw=2):
    """
    Simplifies axis appearance by removing top and right spines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify
    lw : float, default=2
        Line width for remaining spines
    """
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    ax.xaxis.set_tick_params(width=lw)
    ax.yaxis.set_tick_params(width=lw)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left() 
    
    
def plot_axon_native(neuron, r_elec=None, color='g', ax=None, 
                     plot_centers=True, plot_borders=True):
    """
    Plots axon and electrode location in x-y plane.

    Parameters
    ----------
    neuron : brian2.SpatialNeuron
        Neuron object to plot
    r_elec : array-like, optional
        Electrode coordinates
    color : str, default='g'
        Color for axon fill
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    plot_centers : bool, default=True
        Whether to plot segment centers
    plot_borders : bool, default=True
        Whether to plot segment borders

    Returns
    -------
    matplotlib.axes.Axes
        The plotting axes
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
    """
    Scale bar that can be anchored to plots.

    Parameters
    ----------
    transform : matplotlib.transforms.Transform
        Coordinate frame for scale bar
    sizex : float, default=0
        Width of x bar in data units
    sizey : float, default=0
        Height of y bar in data units
    labelx : str, optional
        Label for x bar
    labely : str, optional
        Label for y bar
    loc : int, default=4
        Location code for scale bar placement
    pad : float, default=0.1
        Padding around scale bar
    borderpad : float, default=0.1
        Border padding
    sep : float, default=2
        Separation between labels and bars
    prop : dict, optional
        Font properties
    barcolor : str, default="black"
        Color of scale bar
    barwidth : float, optional
        Width of scale bar lines
    **kwargs : dict
        Additional arguments passed to AnchoredOffsetbox
    """
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
    """
    Adds scale bars to axes matching tick spacing.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add scale bar to
    matchx : bool, default=True
        Match x scale bar to x-axis ticks
    matchy : bool, default=True
        Match y scale bar to y-axis ticks
    hidex : bool, default=True
        Hide x-axis
    hidey : bool, default=True
        Hide y-axis
    **kwargs : dict
        Additional arguments passed to AnchoredScaleBar

    Returns
    -------
    AnchoredScaleBar
        The created scale bar
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

def plot(t, v, r, af, nrn, spk_count, stim_dur, biphasic_ratio, 
         name, suffix, AF0_suffix, idx_start=0,  
         amp_scale=1300, amp_axis_loc=1500, 

         skip_waveform=False,
         save_root=''):
    """
    Creates comprehensive visualization of neuron simulation results.

    Parameters
    ----------
    t : array-like
        Time points
    v : array-like
        Membrane voltages
    r : array-like
        Spatial coordinates
    af : array-like
        Activation function values
    nrn : brian2.SpatialNeuron
        Simulated neuron object
    spk_count : array-like
        Spike counts
    stim_dur : float
        Stimulus duration
    biphasic_ratio : float
        Ratio for biphasic stimulation
    name : str
        Base name for saving
    suffix : str
        Suffix for filename
    AF0_suffix : str
        Activation function suffix
    idx_start : int, default=0
        Starting index for plotting
    amp_scale : float, default=1300
        Amplitude scaling factor
    amp_axis_loc : float, default=1500
        Location of amplitude axis
    skip_waveform : bool, default=False
        Whether to skip waveform plotting
    save_root : str, default=''
        Root directory for saving plots
    """    
    dt = t[1] - t[0]
    #idx_start = 0# int(round((3-2.9)/0.01))+1
    idx_counter = idx_start + int(round(stim_dur/dt))
    idx_end = idx_start + int(round(stim_dur* (1+biphasic_ratio)/dt))
    idx_end_ax = idx_start + int(round(stim_dur* (1+(max(10,biphasic_ratio)))/dt))
    idx_end_ax = idx_start + min(idx_end_ax, .8*len(t))
           
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
    sol = axs['sol'].imshow(v.__array__()*1000, #mV 
                            vmin=-200, vmax=200, cmap='bwr', 
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
    # set_trace()

    # sign = np.sign(float(name.split('_')[0][1:]))
    # val = np.abs(float(name.split('_')[0][1:])) #in mA
    sign = np.sign(float(name.split('_')[0][1:].split(' ')[0]))
    val = np.abs(float(name.split('_')[0][1:].split(' ')[0])) #in mA
    
    if not skip_waveform: 
        # wavefrom axis 
        axs['sol'].plot([idx_start, idx_end_ax*1.15], [amp_axis_loc, amp_axis_loc], color='k', linewidth=0.5) # pseudo x-axis
        # axs['sol'].plot([idx_start, idx_start],[amp_axis_loc, amp_axis_loc+amp_scale*1.5], color='k', linewidth=0.5)

        # scale bar
        axs['sol'].plot([idx_end_ax*1.2, idx_end_ax*1.2], 
                        [amp_axis_loc, amp_axis_loc + amp_scale], 
                        color='k', linewidth=3) # scale bar
        
        axs['sol'].text(idx_end_ax*1.22, amp_axis_loc, f'{val:.2f} mA', 
                        color='k', rotation=90, fontsize=8, 
                        transform=axs['sol'].transData) # scale bar text

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
    axs['af'].plot(af/max(abs(af)), nrn._indices(), label='primary', color='green')

    # spikes
    axs['spk'].plot(spk_count, nrn._indices())
    
    # biphasic plots
    if biphasic_ratio:
        
        if not skip_waveform:
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
        axs['af'].plot(-af/biphasic_ratio/max(abs(af)), nrn._indices(), 
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
    
    axs['af'].set_ylabel('Segment index')

    # x-axis
    axs['af'].set_xlim(-1.2, 1.2)
    # if suffix=='full':
    #     axs['af'].set_xlim(-5, 5)
    # else:
    #     axs['af'].set_xlim(-500,500)

    axs['spk'].set_xlim(-0.2, max(spk_count)+.2)
    if max(spk_count)>1:
        axs['spk'].set_xticks([0,1,max(spk_count)])
    else:
        axs['spk'].set_xticks([0,1])
    # axs['spk'].set_xticklabels(['0','1','2','3'])

    axs['cb'].xaxis.tick_top()

    ticks_times = t[::int(.1*ms//b2.defaultclock.dt)]
    ticks_idx = [i for i,_ in enumerate(t) if t[i] in ticks_times]
    ticks_labels = [str(round(l.__float__(),3)) for l in ticks_times] 
    axs['sol'].set_xticks(ticks_idx)
    axs['sol'].set_xticklabels(ticks_labels)
    
    #tidx = np.linspace(0, len(t) - 1, 17, dtype=int)
    #tidx = np.linspace(0, t[-1], 17, dtype=int)
    #axs['sol'].set_xticks(tidx, np.round(t.__array__()*1000, 1)[tidx])
    #axs['sol'].set_xticklabels(ticks_labels)
    axs['sol'].set_xlabel('Time [ms]')

    # titles
    axs['spk'].set_xlabel('Spike count')
    axs['af'].set_xlabel('Normalized AF\n'+r"$(AF_0= $"+ f"{max(abs(af)):.2f} " + r"$\mathrm{A/m^2}$)")
    # axs['af'].text(.1, 0.9, , 
    #                 color='k', fontsize=8, transform=axs['af'].transAxes)
    axs['cb'].set_title(r'V - V$_\mathrm{rest}$ [mV]')

    # legend
    axs['af'].legend(loc=3)
    plt.tight_layout()
    plt.savefig(save_root +'/'+ name+ '_'+suffix+AF0_suffix+'.png', dpi=200,
                bbox_inches='tight')
    plt.close('all')

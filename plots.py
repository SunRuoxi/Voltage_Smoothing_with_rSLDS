import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rslds.plotting as rplt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set some nice colors
color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "light brown"]


colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")

def plot_most_likely_dynamics(
                              reg, dynamics_distns, colors,
                              xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
                              alpha=0.8,
                              ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    
    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)


    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:D_latent+1]
        F = dynamics_distns[k].A[:, D_latent+1:D_latent+2]
        
        dydt_m = xy.dot(A.T) + b.T  - xy
        
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt_m[zk, 0], dydt_m[zk, 1],
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax


def plot_z_by_class(t, z, K, colors, ax=None, figsize=(3,3), lw=2):
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    points = np.array([t, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    jump_idx = np.array([seg[0][1] != seg[1][1] for seg in segments])
    segments_use = segments[jump_idx==False]
    segments_not = segments[jump_idx==True] # jump

    # Use a boundary norm instead
    cmap = ListedColormap(colors[:K])
    norm = BoundaryNorm(np.array(list(range(0, K+1))) - .5, K+1)
    lc = LineCollection(segments_use, cmap=cmap, norm=norm)
    lc.set_array(z[1:][jump_idx==False])
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    for seg_n in segments_not:
        ax.plot(seg_n[:,0], seg_n[:,1], 'gray',linestyle=':', lw=lw)

    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-0.1, K - 1 + .1)
    ax.set_yticks(list(range(K)))




def make_figure(rslds, zs_rslds, x_rslds):
    

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2, 2)
    
    fp = FontProperties()
    fp.set_weight("bold")
    

    ax3 = fig.add_subplot(gs[0, 0])
    plot_most_likely_dynamics(rslds.trans_distn,
                              rslds.dynamics_distns, colors,
                              xlim=(-4, 8), ylim=(-.5, .5),
                              ax=ax3)
        
                              
    # Overlay a partial trajectory
    rplt.plot_trajectory(zs_rslds[-1], x_rslds, ax=ax3, ls="-")
    ax3.set_title("Inferred Dynamics (rSLDS)")


    # Plot samples of discrete state sequence
    ax4 = fig.add_subplot(gs[1,0])
    rplt.plot_z_samples(rslds.num_states, zs_rslds, plt_slice=(0,x_rslds.shape[0]), ax=ax4)
    ax4.set_title("Discrete State Samples")



    ax5 = fig.add_subplot(gs[0, 1])
    plot_input_dynamics(rslds.trans_distn,
                      rslds.dynamics_distns,
                      xlim=(-4, 8), ylim=(-4, 10),
                      ax=ax5)

    ax6 = fig.add_subplot(gs[1, 1])
    plot_other_compartments_dynamics(rslds.trans_distn,
                                   rslds.dynamics_distns,
                                   xlim=(-4, 8), ylim=(-4, 10),
                                   ax=ax6)
                              
    plt.tight_layout()


def plot_input_dynamics(
                        reg, dynamics_distns,
                        xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
                        alpha=0.8,
                        ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    
    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        F = dynamics_distns[k].A[:, D_latent+1:D_latent+2]
        dydt = 10 * F
        
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt[0,0], dydt[1,0],
                      scale=0.1, scale_units="inches",
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$V$')
    ax.set_ylabel('$\Delta V$')
    #ax.title('scale for V per 10mA of')

    plt.tight_layout()
    
    return ax

def plot_full_tree(paths, compartments_byid, select_compartments, directory):
    

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(20, 10))
    axes[0].set_aspect('equal', 'box-forced')
    axes[1].set_aspect('equal', 'box-forced')
    
    
    # Make a line drawing of x-y and y-z views
    
    for i, path in enumerate(paths):
        for j in range(len(path)-1):
            start_id = path[j]
            end_id = path[j+1]
            n = compartments_byid[start_id]
            c = compartments_byid[end_id]
            
            axes[0].plot([n['x'], c['x']], [n['y'], c['y']], '.-', color=colors[i%len(colors)])
            axes[1].plot([n['z'], c['z']], [n['y'], c['y']], '.-', color=colors[i%len(colors)])

    # cut dendrite markers
    dm = [ compartments_byid[m] for m in select_compartments]
    axes[0].scatter([m['x'] for m in dm], [m['y'] for m in dm], color='k', s=100, marker='o')
    axes[1].scatter([m['z'] for m in dm], [m['y'] for m in dm], color='k', s=100, marker='o')


    axes[0].set_ylabel('y')
    axes[0].set_xlabel('x')
    axes[1].set_xlabel('z')
    #plt.show()
    plt.savefig(directory+'full_tree.pdf',bbox_inches='tight')


def plot_desampled_tree(path_compartments, compartments_byid, directory):
    
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(20, 40))
    axes[0].set_aspect('equal', 'box-forced')
    axes[1].set_aspect('equal', 'box-forced')
    
    # Make a line drawing of x-y and y-z views
    
    for i, path in enumerate(path_compartments):
        for j in range(len(path)-1):
            start_id = path[j]
            end_id = path[j+1]
            n = compartments_byid[start_id]
            c = compartments_byid[end_id]
            
            axes[0].plot([n['x'], c['x']], [n['y'], c['y']], '.-', color=colors[i%len(colors)])
            axes[1].plot([n['z'], c['z']], [n['y'], c['y']], '.-', color=colors[i%len(colors)])

    axes[0].set_ylabel('y')
    axes[0].set_xlabel('x')
    axes[1].set_xlabel('z')
    plt.savefig(directory+'desampled_tree.pdf',bbox_inches='tight')

def plot_subtree(new_id, directory, paths, compartments_byid, p_start = 0, p_end = 7, axes=None, sz=12, point_sz=10):
    
    if axes is None:
        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(20, 20))
    
    # Make a line drawing of x-y and y-z views
    fsz1 = 10
    fsz2 = 10
    lw = 3
    

    p_end =  min(p_end, len(paths))

    for i in range(p_start,  p_end):
        path = paths[i]
        for j in range(len(path)-1):
            start_id = path[j]
            end_id = path[j+1]
            n = compartments_byid[start_id]
            c = compartments_byid[end_id]
            
            axes[0].plot([n['x'], c['x']], [n['y'], c['y']], '-', color=colors[i%len(colors)], linewidth=lw)
            axes[1].plot([n['z'], c['z']], [n['y'], c['y']], '-', color=colors[i%len(colors)], linewidth=lw)

    for path in paths[p_start:p_end]:
        for i in path:
            c = compartments_byid[i]
            if c['id'] in new_id:
                axes[0].text(c['x']+1, c['y']+1, str(new_id[c['id']]), fontweight='bold',fontsize=point_sz)
                axes[1].text(c['z']+1, c['y'], str(new_id[c['id']]), fontweight='bold', fontsize=point_sz)



    c0 = compartments_byid[0]
    axes[0].annotate('', xy=(c0['x'], c0['y']), xytext=(c0['x'], c0['y']-10),
                     arrowprops=dict(facecolor='red', shrink=0.05), fontweight='bold', fontsize=fsz1)

    axes[1].annotate('', xy=(c0['z'], c0['y']), xytext=(c0['z'], c0['y']-10),
                     arrowprops=dict(facecolor='red', shrink=0.05), fontweight='bold', fontsize=fsz1)


    # cut dendrite markers

    dm = [ compartments_byid[m] for m_p in paths[p_start:p_end] for m in m_p if m in new_id]
    axes[0].scatter([m['x'] for m in dm], [m['y'] for m in dm], color='k', s=100, marker='o')
    axes[1].scatter([m['z'] for m in dm], [m['y'] for m in dm], color='k', s=100, marker='o')

    axes[0].set_ylabel('y', fontweight='bold', fontsize=sz, rotation=0)
    axes[0].set_xlabel('x', fontweight='bold', fontsize=sz)
    axes[1].set_ylabel('y', fontweight='bold', fontsize=sz, rotation=0)
    axes[1].set_xlabel('z', fontweight='bold', fontsize=sz)


    plt.savefig(directory+'subtree-start='+str(p_start) + ', end=' + str(p_end) +'.pdf')

def plot_inferred_bypath(paths_filtered, num_compartments, per_page, folder, t, x_smpl_compartments, z_smpls_compartments, I_inj_values, Vs_obs, Vs_true, new_id):

    toplot = slice(100,1000)

    for start in range(0, len(paths_filtered), per_page):
        plt.close('all')
        end = min(start + per_page, len(paths_filtered))
        len_sum = 0
        for i in range(start, end):
            len_sum += len(paths_filtered[i])
    
        plt.close('all')
        if len_sum > 40:
            fig = plt.figure(figsize=(7, 1 + 3 * (len_sum)))
        else:
            fig = plt.figure(figsize=(7, 1 + 4 * (len_sum)))
        nplot = (len_sum) * 2
        
        plot_idx = 1
        for j, path in enumerate(paths_filtered[start:end]):
            for idx in path:
                print(idx)
                c = new_id[idx]
                
                plt.subplot(nplot, 1, plot_idx)
                plt.plot(t[toplot], Vs_obs[toplot,c], 'black', label='obs')
                plt.plot(t[toplot], x_smpl_compartments[c][-1][toplot,0], '--', label='inferred x', color=colors[(start+j)%len(colors)]);
                plt.plot(t[toplot], Vs_true[toplot,c], 'r', label='True')
                 
                plt.legend()
               
                plt.xticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100 ms'])
                path_convert = [new_id[p] for p in path]
                plt.title('PATH='+str(path_convert)+'. compartment '+str(c)+': inferred x')
                plot_idx += 1
                
                plt.subplot(nplot, 1, plot_idx)
                plt.plot(t[toplot], z_smpls_compartments[c][-1][-1,toplot], color=colors[(start+j)%len(colors)], label='inferred z');

                plt.title('PATH='+str(path_convert)+'. compartment '+str(c)+': inferred z')
                
                plot_idx += 1
                plt.legend()
        plt.xlabel('t (ms)')
        plt.tight_layout()
        fig.savefig(folder+'new-train-compare-bypath-start=' + str(start) + '.pdf')

def plot_latent(rslds, zs_rslds, x_rslds):
    
 
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2, 2)
    
    fp = FontProperties()
    fp.set_weight("bold")
    
    ax3 = fig.add_subplot(gs[0, 0])
    plot_most_likely_dynamics(rslds.trans_distn,
                              rslds.dynamics_distns,
                              xlim=(-4, 8), ylim=(-.5, .5),
                              ax=ax3)
        
        
        
    # Overlay a partial trajectory
    rplt.plot_trajectory(zs_rslds[-1], x_rslds, ax=ax3, ls="-")
    ax3.set_title("Inferred Dynamics (rSLDS)")

    # Plot samples of discrete state sequence
    ax4 = fig.add_subplot(gs[1,0])
    rplt.plot_z_samples(rslds.num_states, zs_rslds, plt_slice=(0,x_rslds.shape[0]), ax=ax4)
    ax4.set_title("Discrete State Samples")


    ax5 = fig.add_subplot(gs[0, 1])
    plot_input_dynamics(rslds.trans_distn,
                  rslds.dynamics_distns,
                  xlim=(-4, 8), ylim=(-4, 10),
                  ax=ax5)

    ax6 = fig.add_subplot(gs[1, 1])
    plot_other_compartments_dynamics(rslds.trans_distn,
                               rslds.dynamics_distns,
                               xlim=(-4, 8), ylim=(-4, 10),
                               ax=ax6)
        
    plt.tight_layout()

def plot_most_likely_dynamics(
                              reg, dynamics_distns,
                              xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
                              alpha=0.8,
                              ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    
    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:D_latent+1]
        F = dynamics_distns[k].A[:, D_latent+1:D_latent+2]
        
        dydt_m = xy.dot(A.T) + b.T  - xy
        
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt_m[zk, 0], dydt_m[zk, 1],
                      scale=10, scale_units="inches",
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$V$')
    ax.set_ylabel('$\Delta V$')

    plt.tight_layout()

    return ax

def plot_input_dynamics(
                        reg, dynamics_distns,
                        xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
                        alpha=0.8,
                        ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    
    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        F = dynamics_distns[k].A[:, D_latent+1:D_latent+2]
        dydt = 10 * F
        
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt[0,0], dydt[1,0],
                      scale=0.1, scale_units="inches",
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$V$')
    ax.set_ylabel('$\Delta V$')

    plt.tight_layout()
    
    return ax

def plot_other_compartments_dynamics(
                                     reg, dynamics_distns,
                                     xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
                                     alpha=0.8,
                                     ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))
    
    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        F = dynamics_distns[k].A[:, D_latent+2:D_latent+3]
        dydt = 10 * F
        
        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt[0,0], dydt[1,0],
                      scale=0.1, scale_units="inches",
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$V$')
    ax.set_ylabel('$\Delta V$')


    plt.tight_layout()
    
    return ax





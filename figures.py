import numpy as np
import scipy as sp
import seaborn as sns
import numpy.random as npr
import matplotlib.pyplot as plt
import rslds.plotting as rplt
import matplotlib.gridspec as gridspec
from NIPS_single_cmpt import run_real_data
from matplotlib.font_manager import FontProperties
from NIPS_plots import plot_most_likely_dynamics, plot_z_by_class


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
               "dark brown"]

colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")

def figure2_single_compartment():
    seed = 0
    npr.seed(seed)
    # run rslds
    mse, x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values = run_real_data(K=3, noise_std=.2, sigmasq_value=.05, penalty=.1)
    # plot rslds
    plot_single_compartment('figure2_single_compartment.pdf', x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values)
    
def plot_single_compartment(save_figure_name, x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values):
    # generate voltage using recurrent SLDS
    rslds_x_gen_hist = []
    rslds_z_gen_hist = []
    
    for seed1 in range(5, -1, -1):
        npr.seed(seed1)
        
        T_sim = min(50000, len(I_inj_values))
        
        inputs = np.column_stack((np.ones((min(T_sim, len(I_inj_values)), 1)), I_inj_values[:T_sim]))
        rslds_y_gen, rslds_x_gen, rslds_z_gen = rslds.generate(T=T_sim, inputs=inputs)
        
        rslds_x_gen_hist.append(rslds_x_gen)
        rslds_z_gen_hist.append(rslds_z_gen)
    
    V_MEAN = -38.98491
    V_STD = 12.504922
    
    x_std = np.std(x_smpls[-200:,:,0], axis=0)  * V_STD
    x_mean0 = np.mean(x_smpls[-200:,:,0], axis=0) *V_STD + V_MEAN
    x_mean1 = np.mean(x_smpls[-200:,:,1], axis=0)
    x_mean = np.concatenate((x_mean0[:,None], x_mean1[:,None]), axis=1)
    z_mean = np.mean(z_smpls[-200:,:], axis=0)
    
    fig = plt.figure(figsize=(15,15)) # good
    gs = gridspec.GridSpec(6, 4, figure=fig)
    
    
    dt = t[1] - t[0]
    
    fp = FontProperties()
    fp.set_weight("bold")
    sz = 20
    sz1 = 20
    title_sz = 20
    legend_sz = 20
    dx = 0.025
    dy = 0.035
    
    start, end = 1000, 1550
    select = slice(start,end)
    t = sp.arange(0, (end-start)*dt, dt)
    
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, V_obs[select]*V_STD + V_MEAN, 'k-', label='Observed voltage', lw=3);
    plt.figtext(.025-dx, 1-.07+dy, '(a)', fontproperties=fp, fontsize=sz)
    ax1.set_title('Observed Voltage & Inferred Continuous Latent State X1', fontsize=title_sz, fontweight="bold")
    
    ax1.plot(t, x_mean0[select], 'b--', lw=2);
    ax1.set_xticks([], [])
    ax1.fill_between(t, x_mean0[select]-x_std[select], x_mean0[select]+x_std[select], alpha=.7, label='CI', color='c', lw=3)
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, shadow=True, ncol=2, prop={'size': legend_sz})
    
    
    plt.ylabel('V (mV)', fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    plt.yticks(fontsize=legend_sz, fontweight="bold")
    
    
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, x_mean1[select], color=colors[7], lw=3)
    ax2.set_xticks([], []); ax2.set_yticks([], [])
    plt.figtext(.025-dx, 1-  0.16*1 - .07+dy, '(b)', fontproperties=fp, fontsize=sz)
    ax2.set_title('Inferred Continuous Latent State X2', fontsize=title_sz, fontweight="bold")
    plt.yticks(fontsize=legend_sz, fontweight="bold")
    
    ax3 = fig.add_subplot(gs[2, :])
    z_smpl = z_mean[select]
    plot_z_by_class(t, z_smpl, K, colors, ax3, lw=3)
    plt.figtext(.025-dx, 1-  0.16*2 - .07+dy, '(c)', fontproperties=fp, fontsize=sz)
    ax3.set_title('Inferred Discrete Latent State Z', fontsize=title_sz, fontweight="bold")
    plt.xticks(fontsize=legend_sz, fontweight="bold")
    plt.yticks(fontsize=legend_sz, fontweight="bold")
    ax3.set_xlabel('t (ms)', fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    
    
    ax4 = fig.add_subplot(gs[3:5, :2])
    plot_most_likely_dynamics(rslds.trans_distn,
                              rslds.dynamics_distns,
                              colors,
                              xlim=(-2, 6), ylim=(-.3, .3),
                              ax=ax4)
    x_smpl_rescale =  x_smpl.copy()
    ax4.set_xticks([], []); ax4.set_yticks([], [])
    x_smpl_rescale[:,0] = x_smpl_rescale[:,0]*V_STD + V_MEAN

    rplt.plot_trajectory(z_smpls[-1,:], x_smpl, ax=ax4, ls="-")
    ax4.set_title("Inferred Dynamics (rSLDS)", fontsize=title_sz, fontweight="bold")
    ax4.set_ylabel('X', fontsize=title_sz, fontweight="bold")
    ax4.set_xlabel('V', fontsize=title_sz, fontweight="bold")
    plt.figtext( -dx + .025, 1-  0.16*3 - .07+dy, '(d)', fontproperties=fp, fontsize=sz)
    plt.xticks(fontsize=legend_sz, fontweight="bold")


    ax5 = fig.add_subplot(gs[3:5, 2:4])
    rslds_x_gen_rescale = rslds_x_gen.copy()
    rslds_x_gen_rescale[:,0] = rslds_x_gen_rescale[:,0] *V_STD + V_MEAN
    rplt.plot_trajectory(rslds_z_gen[-1000:], rslds_x_gen_rescale[-1000:], ls="-", ax=ax5)

    ax5.set_xlim((-60, 40))
    ax5.set_ylim((-.3,.3))
    ax5.set_ylabel('X', fontsize=title_sz, fontweight="bold")
    ax5.set_xlabel('V', fontsize=title_sz, fontweight="bold")
    ax5.set_xticks([], []); ax5.set_yticks([], [])
    ax5.set_xticks([], []); ax5.set_yticks([], [])
    plt.xticks(fontsize=legend_sz, fontweight="bold")
    plt.yticks(fontsize=legend_sz, fontweight="bold")
    plt.figtext(+.6+.025-dx, 1- 0.16*3 -0.07+dy, '(e)', fontproperties=fp, fontsize=sz)
    ax5.set_title('Generated States', fontsize=title_sz, fontweight="bold")
    plt.xticks(fontsize=legend_sz, fontweight="bold")

    ax6 = fig.add_subplot(gs[5, :])
    start1, end1 = 5100, 5650
    select1 = slice(start1, end1)
    t1 = sp.arange(0, (end1 - start1)*dt, dt)
    ax6.plot(t1, rslds_x_gen_hist[0][select1,0]*V_STD + V_MEAN, color='green', label='Sample 1', lw=3)
    ax6.plot(t1, rslds_x_gen_hist[1][select1,0]*V_STD + V_MEAN, color='orange', label='Sample 2', lw=3)
    ax6.plot(t1, rslds_x_gen_hist[2][select1,0]*V_STD + V_MEAN, color='purple', label='Sample 3', lw=3)
    ax6.set_xlabel('t (ms)', fontsize=legend_sz, fontweight="bold")
    legend_sz_s = 10
    plt.legend(prop={'size': legend_sz_s, 'weight':'bold'})

    plt.figtext( .025-dx, 1-  0.16*5 - .075+dy, '(f)', fontproperties=fp, fontsize=sz)
    ax6.set_title('Samples Drawn from Trained rSLDS', fontsize=title_sz, fontweight="bold")
    ax6.set_ylabel('V (mV)', fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    ax6.tick_params(axis='both', which='minor', labelsize=legend_sz)
    plt.xticks(fontsize=legend_sz, fontweight="bold")
    plt.yticks(fontsize=legend_sz, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_figure_name, bbox_inches = 'tight')

def run_comparison_realdata_noise():
    
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    sigmasq_values = [0.01, 0.025, 0.1, 0.15, 0.22, 0.3, 0.63]
    MSE_K3 = []
    K = 3
    for i in range(len(noises)):
        noise_std = noises[i]
        penalty = .05
        seed = 0
        npr.seed(seed)
        print("Seed was:", seed)
        sigmasq_value = sigmasq_values[i]
        mse, x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values = run_real_data(K, noise_std, sigmasq_value, penalty)
        MSE_K3.append(mse)
        
        
    MSE_K1 = []
    K = 1
 
    for i in range(len(noises)):
        seed = 0
        npr.seed(seed)
 
        noise_std = noises[i]
        penalty = .02
        
        sigmasq_value = sigmasq_values[i]
        mse, x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values = run_real_data(K, noise_std, sigmasq_value, penalty)
        MSE_K1.append(mse)
        
 

    
    return MSE_K3, MSE_K1, noises



def figure3_compare_with_baseline_different_noise():


    seed = 0
    npr.seed(seed)
 
    mse3, x_smpl3, V_true3, V_obs3, _, _, _, _, _, _ = run_real_data(K=3, noise_std=0.7, sigmasq_value=0.63, penalty=0.05)
    mse1, x_smpl1, V_true1, V_obs1, _, _, _, _, _, _ = run_real_data(K=1, noise_std=0.7, sigmasq_value=0.65, penalty=0.02)
    
    
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(4,8, figure=fig)
    fp = FontProperties()
    fp.set_weight("bold")
    sz = 20
    sz1 = 20
    title_sz = 20
    legend_sz = 15
    
    dt = 0.1
    t = sp.arange(0, len(V_true3), 0.1)
    fp = FontProperties()
    fp.set_weight("bold")
    
    ax1 = fig.add_subplot(gs[:2, 0:6])
    select = slice(200,1200)
    lw = 1.7
    
    #from sklearn.metrics import mean_squared_error
    V_MEAN = -38.98491
    V_STD = 12.504922
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    title_sz = 20
    ax1.plot(t[select], V_obs1[select]*V_STD +V_MEAN , 'k-', label='Observed noisy voltage');
    ax1.set_ylabel('Voltage (mV)',rotation=90, fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    ax1.plot(t[select], x_smpl1[select,0]*V_STD +V_MEAN,'y-', label='Inferred voltage: K=1');
    ax1.plot(t[select], V_true1[select]*V_STD +V_MEAN, 'b--', lw=lw, label='True');
    ax1.set_title('Denoised voltage K=1 (noise level = 0.7)', fontsize=title_sz, fontweight="bold")
    
    plt.legend(loc=1)
    
    
    ax3 = fig.add_subplot(gs[2:4, 0:6])
 
    
    ax3.plot(t[select], V_obs3[select]*V_STD +V_MEAN, 'k-', label='Observed noisy voltage');
    ax3.set_ylabel('Voltage (mV)',rotation=90, fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    ax3.set_xlabel('t (ms)', fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    ax3.plot(t[select], x_smpl3[select,0]*V_STD +V_MEAN,'r-', label='Inferred voltage: K=3');
    ax3.plot(t[select], V_true3[select]*V_STD +V_MEAN, 'b--', lw=lw, label='True');
    ax3.set_title('K=3', fontsize=title_sz, fontweight="bold")
    
    plt.legend(loc=1)
    

    MSE_l1_tf = np.array([0.00419674, 0.01182731, 0.02232965, 0.03580776, 0.05198877,
                          0.07035884, 0.0909741])
    
    MSE_K3, MSE_K1, noises = run_comparison_realdata_noise()

    ax2 = fig.add_subplot(gs[1:3, 6:8])
    ax2.plot(noises, MSE_K3, 'ro-', label='K=3')
    ax2.plot(noises, MSE_K1, 'yo-',label='K=1')
    ax2.plot(noises, MSE_l1_tf, 'go-',label='L1TF')
    ax2.set_xlabel('Noise level (mV)', fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    ax2.set_ylabel('MSE (mV/time point)', fontproperties=fp, fontsize=legend_sz, fontweight="bold")
    ax2.set_title('MSE', fontsize=title_sz, fontweight="bold")
    plt.legend()
   
    plt.tight_layout()
    plt.savefig('figure3_compare_with_baseline_different_noise.pdf')



def compare(K):
    
 
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    sigmasq_values3 = [0.01, 0.025, 0.1, 0.15, 0.22, 0.3, 0.63]
    sigmasq_values4 = [0.01, 0.025, 0.1, 0.15, 0.3, 0.5, 0.63]
    sigmasq_values_all = {3:sigmasq_values3, 4:sigmasq_values4, }
    
    MSE_hist = []
    outputs_hist = []

    for i in range(len(noises)):
        noise_std = noises[i]
        penalty = .05
        seed = 0
        npr.seed(seed)
        print("Seed was:", seed)

        sigmasq_value = sigmasq_values[i]
        mse, x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values = run_real_data(K, noise_std, sigmasq_value, penalty)
        # save results
        outputs_hist.append((x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values))
        MSE_hist.append(mse)
        # plot
        save_figure_name = 'noise'+str(noise_std)+'_K'+str(K)
        plot_single_compartment(save_figure_name+'.pdf', x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values)
        plt.figure(); plt.plot(lps); plt.title('lps'); plt.xlabel('Iterations'); plt.savefig(save_figure_name+'_lps.pdf')
    
    return MSE_hist, outputs_hist






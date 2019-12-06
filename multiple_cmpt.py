import os
from os import path
import numpy as np
import scipy as sp
from tqdm import tqdm
import numpy.random as npr
import matplotlib.pyplot as plt
import rslds.plotting as rplt
from plots import plot_full_tree, plot_desampled_tree, plot_subtree, plot_inferred_bypath, plot_latent, plot_most_likely_dynamics
from utils import make_roslds_mtpl, run_gibbs_sampler, MultiCompartmentHodgkinHuxley, load_realdata_morphology

def coordinate_gibbs_sampler_mtpl(rslds_compartments, network_rm_diag, Vs_obs, num_compartments):
    Vs_inferred = Vs_obs.copy()

    z_smpls_compartments = [[] for _ in range(num_compartments)]
    x_smpl_compartments = [[] for _ in range(num_compartments)]

    for nter in tqdm(range(N_samples)):

        for j in range(num_compartments):
            # Run Gibbs sampler on compartment j, holding the inputs
            # to this compartment (i.e. the voltage from connected compartments) fixed.
            lps, z_smpls, x_smpl = run_gibbs_sampler(rslds_compartments[j], N_samples=N_samples_compartments)
            x_smpl_compartments[j].append(x_smpl)
            z_smpls_compartments[j].append(z_smpls)

            # Now update the voltage of this compartment in Vs_inferred
            Vs_inferred[:, j] = x_smpl[:,0]

            # Compute the new "inputs" to all compartments using the updated Vs
            # network_rm_diag: receiving compartment x sending compartment
            # Vs_inferred: time x sending_compartment
            # V_compartments: receiving compartment x time
            V_compartments = network_rm_diag.dot(Vs_inferred.T)

            # Now update the inputs to each compartment
            # The second input is the summed, weighted voltage from other compartments
            for c in range(num_compartments):
                assert len(rslds_compartments[c].states_list) == 1
                rslds_compartments[c].states_list[0].inputs[:,2] =  V_compartments[c,:]
    return rslds_compartments, x_smpl_compartments, z_smpls_compartments


def run_multiple_realshape(sigmasq_value, penalty, load_data = False):
    # sigmasq_value and penalty will need to be tuned for best performance.

    directory = './results/'

    if not os.path.exists(directory):
        os.mkdir(directory)

    # load real morphology data
    desample_tree_factor = 5
    network, new_id, compartments_byid, paths, select_compartments, path_compartments, path_filtered, compartments, morphology = load_realdata_morphology(cell_id = 464212183, desample_tree_factor=desample_tree_factor)
    # plot full tree
    plot_full_tree(paths, compartments_byid, select_compartments, directory)
    # plot desampled tree
    plot_desampled_tree(path_compartments, compartments_byid, directory)
    # plot subtree
    per_page = 10
    for j in range(0, len(paths), per_page):
        plot_subtree(new_id, directory, paths, compartments_byid, p_start = j, p_end = j + per_page)


    num_compartments = network.shape[0]
    conductance = 5
    network = network * conductance
    plt.close('all'); plt.figure(); plt.imshow(network[:200,:200]); plt.title('dist'); plt.colorbar(); plt.savefig(directory+'dist, conductance='+str(conductance) +'.png')
    network_offdiag = network - np.diag(np.diag(network))
    np.fill_diagonal(network,  -np.sum(network_offdiag, axis=1))
    plt.figure(); plt.imshow(network[:200,:200]); plt.title('dist'); plt.colorbar(); plt.savefig(directory+'dist, conductance='+str(conductance) +'-modified.png')

    seed =  0
    npr.seed(seed)


    noise_std = 0.5  # noise standard deviation level
    compartment_hypers = [{} for _ in range(num_compartments)]
    model = MultiCompartmentHodgkinHuxley(num_compartments, compartment_hypers, network)
    dt = 0.1
    t_train = 100 # training data
    t = sp.arange(0, t_train, dt)

    # simulate e
    save_name = 'HHsimulation_'+str(noise_std)
    if not load_data:
        # new data
        Vs, ms, hs, ns, I_inj_values, Vs_true, Vs_obs = model.simulate(t, noise_std=noise_std)
        np.savez(directory + save_name, Vs=Vs, I_inj_values=I_inj_values, Vs_true=Vs_true, Vs_obs=Vs_obs, ms=ms, ns=ns, hs=hs)
    else:
        # load
        print('Load previous data!')
        npzfile = np.load(directory + save_name + '.npz')
        Vs = npzfile['Vs']
        I_inj_values = npzfile['I_inj_values']
        Vs_true = npzfile['Vs_true']
        Vs_obs = npzfile['Vs_obs']


    K = 3
    D_latent= 2
    N_samples_compartments = 10
    N_samples = 100

    # to save
    str_name = 'UPDATE' + ', seed='+str(seed)+ ', num_compart=' + str(num_compartments) + ', K=' + str(K) + ', sigmasq_value='+ str(sigmasq_value) + ', penalty=' + str(penalty) +', Niter=' + str(N_samples)+ ', noise_std=' +str(noise_std)+ ', t_train=' + str(t_train) + ', conductance=' + str(conductance) + '-newini' + ', N_samples_compartments='+str(N_samples_compartments)


    file = directory + str_name

    # initialize a series of rslds
    rslds_compartments = []
    V_compartments = network.dot(Vs_obs.T)
    for j in range(num_compartments):

        rslds = make_roslds_mtpl(Vs_obs[:,j], I_inj_values[:,j], V_compartments[j,:], K=K, D_latent=D_latent, sigmasq_value=sigmasq_value, penalty=penalty)

        rslds_compartments.append(rslds)

    # gibbs sampler multiple components
    Vs_inferred = Vs_obs.copy()
    network_rm_diag = network.copy()
    np.fill_diagonal(network_rm_diag, 0) # only consider off diagonal effect; other compartments

    # coordinate gibbs sampler
    rslds_compartments, x_smpl_compartments, z_smpls_compartments = coordinate_gibbs_sampler_mtpl(rslds_compartments, network_rm_diag, Vs_obs, num_compartments)

    # plot per compartment voltage
    per_page = 1
    plt.close('all')
    plot_inferred_bypath(path_filtered, num_compartments, per_page, file, t, x_smpl_compartments, z_smpls_compartments, I_inj_values, Vs_obs, Vs_true, new_id)

    # plot per compartment latent space
    for c in range(num_compartments):
        plt.close('all')
        plot_latent(rslds_compartments[c], z_smpls_compartments[c][-1], x_smpl_compartments[c][-1])
        plt.savefig(file+'cmpt'+str(c)+'-latent.pdf')

    # save results
    save_name = 'trial'+str(noise_std)
    np.savez(directory + save_name, x_smpl_compartments=x_smpl_compartments,  z_smpls_compartments=z_smpls_compartments,  t= t,  I_inj_value = I_inj_values, Vs_obs=Vs_obs, Vs_true=Vs_true, I_inj_values =I_inj_values)
    np.savez(directory + save_name + '_tree', network=network, new_id=new_id, compartments_byid=compartments_byid, paths=paths, select_compartments=select_compartments, path_compartments=path_compartments, path_filtered=path_filtered, compartments=compartments, morphology=morphology)

    return rslds_compartments, x_smpl_compartments, z_smpls_compartments, x_gen_compartments_list, z_gen_compartments_list

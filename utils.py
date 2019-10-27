import numpy as np
import scipy as sp
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.random as npr
from scipy.integrate import odeint
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from pybasicbayes.distributions import Gaussian, Regression, DiagonalRegression
from rslds.models import PGRecurrentOnlySLDS

class HodgkinHuxley(object):
    
    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.387):
        """
            Single compartment Hodgkin-Huxley model.
            """
        self.C_m = C_m      # membrane capacitance, in uF/cm^2
        self.g_Na = g_Na    # Sodium (Na) maximum conductances, in mS/cm^2
        self.g_K = g_K      # Postassium (K) maximum conductances, in mS/cm^2
        self.g_L = g_L      # Leak maximum conductances, in mS/cm^2
        self.E_Na = E_Na    # Sodium (Na) Nernst reversal potentials, in mV
        self.E_K = E_K      # Postassium (K) Nernst reversal potentials, in mV
        self.E_L = E_L      # Leak Nernst reversal potentials, in mV
    
    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))
    
    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)
    
    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)
    
    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
    
    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))
    
    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)
    
    def I_Na(self, V, m, h):
        """
            Membrane current (in uA/cm^2)
            Sodium (Na = element name)
            |  :param V:
            |  :param m:
            |  :param h:
            |  :return:
            """
        return self.g_Na * m**3 * h * (V - self.E_Na)
    
    def I_K(self, V, n):
        """
            Membrane current (in uA/cm^2)
            Potassium (K = element name)
            |  :param V:
            |  :param n:
            |  :return:
            """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
            Membrane current (in uA/cm^2)
            Leak
            |  :param V:
            |  :param h:
            |  :return:
            """
        return self.g_L * (V - self.E_L)
    
    def dynamics(self, X, t, I_inj):
        """
            Integrate
            |  :param X:
            |  :param t:
            |  :return: calculate membrane potential & activation variables
            """
        V, m, h, n = X
        
        dVdt = (I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt
    
    def simulate(self, t, X0=(-65, 0.05, 0.6, 0.32), I_inj=None):
        """
            Main demo for the Hodgkin Huxley neuron model
            :param ts: time points to simulate.
            """
        # Set the injected current function
        I_inj = I_inj if I_inj is not None else lambda t: 35
        
        # Specify the dynamics function
        dynamics = lambda X, t, I_inj: self.dynamics(X, t, I_inj)
        
        # Run the implicit Euler integration
        X = odeint(dynamics, X0, t, args=(I_inj,))
        
        # Extract the state variables
        V = X[:,0]  # voltage
        m = X[:,1]  # sodium channel activation
        h = X[:,2]  # sodium channel inactivation
        n = X[:,3]  # potassium channel activation
        
        # Compute the currents as a function of state variables
        I_Na = self.I_Na(V, m, h)
        I_K = self.I_K(V, n)
        I_L = self.I_L(V)
        
        # Get injected current at each time point
        I_inj_values = np.array([I_inj(ti) for ti in t])
        
        # Return the simulated states and currents
        return V, m, h, n, I_inj_values
    
    def plot_simulation(self, t, V, m, h, n, I_inj_values):
        plt.figure()
        
        plt.subplot(4,1,1)
        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(t, V, 'k')
        plt.ylabel('V (mV)')
        
        plt.subplot(4,1,2)
        plt.plot(t, self.I_Na(V, m, h), 'c', label='$I_{Na}$')
        plt.plot(t, self.I_K(V, n), 'y', label='$I_{K}$')
        plt.plot(t, self.I_L(V), 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()
        
        plt.subplot(4,1,3)
        plt.plot(t, m, 'r', label='m')
        plt.plot(t, h, 'g', label='h')
        plt.plot(t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()
        
        plt.subplot(4,1,4)
        plt.plot(t, I_inj_values, 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 40)


class MultiCompartmentHodgkinHuxley(object):
    
    def __init__(self, num_compartments, compartment_hypers, network):
        assert num_compartments > 0
        assert len(compartment_hypers) == num_compartments
        self.num_compartments = num_compartments
        self.compartments = [HodgkinHuxley(**hypers) for hypers in compartment_hypers]
        
        assert network.shape == (num_compartments, num_compartments)
        network_offdiag = network - np.diag(np.diag(network))
        assert np.all(network_offdiag) >= 0, "All offdiagonal entries must be non-negative"
        assert np.allclose(np.diag(network), -np.sum(network_offdiag, axis=1)), "Diagonal must equal negative sum of offdiagonals"
        assert np.allclose(network - network.T, 0), "Network must be symmetric"
        self.network = network
    
    def dynamics(self, X, t, I_inj):
        """
            X is an array of length 4 * num_compartments. Each compartment has (V, m, h, n)
            t is the time index
            I is an array of length num_compartments, specifying input current to each compartment.
            """
        assert X.shape == (4 * self.num_compartments,)
        # I = I_inj(t)
        # assert I.shape == (self.num_compartments,)
        # dVdt = (I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        # dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        # dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        # dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        
        dynamics = np.zeros(4 * self.num_compartments)
        for c in range(self.num_compartments):
            Xc = X[c*4 : (c+1) * 4]
            Ic = lambda t: I_inj(t)[c]
            dVdt_c, dmdt_c, dhdt_c, dndt_c = self.compartments[c].dynamics(Xc, t, Ic)
            
            # Add the dVdt from cable equation
            V = X[::4]
            dVdt_c += self.network[c].dot(V) / self.compartments[c].C_m
            
            # Insert this compartment's dynamics into the global dynamics vector
            for i, dxdt in enumerate([dVdt_c, dmdt_c, dhdt_c, dndt_c]):
                dynamics[c*4 + i] = dxdt
        
        return dynamics
                

    def simulate(self, t, X0=np.array((-65, 0.05, 0.6, 0.32)), I_inj=None, noise_std = 1e-4):
        
        """
        Main demo for the Hodgkin Huxley neuron model
        :param ts: time points to simulate.
        """
        # Set the injected current function
        I_inj = I_inj if I_inj is not None else \
        lambda t: np.concatenate(([35], np.zeros(self.num_compartments-1)))
            
        # Specify the dynamics function
        dynamics = lambda X, t, I_inj: self.dynamics(X, t, I_inj)
        
        # Run the implicit Euler integration
        sigma0 = np.array([3, .01, .1, .05])
        X0 = np.concatenate([X0 + sigma0 * npr.randn(4)
                             for _ in range(self.num_compartments)])
        X = odeint(dynamics, X0, t, args=(I_inj,))
                                         
        # Extract the state variables
        Vs = X[:,0::4]  # voltage
        ms = X[:,1::4]  # sodium channel activation
        hs = X[:,2::4]  # sodium channel inactivation
        ns = X[:,3::4]  # potassium channel activation
         
        # Compute the currents as a function of state variables
        I_Nas = [c.I_Na(V, m, h) for c, V, m, h, n in zip(self.compartments, Vs, ms, hs, ns)]
        I_Ks = [c.I_K(V, n) for c, V, m, h, n in zip(self.compartments, Vs, ms, hs, ns)]
        I_Ls = [c.I_L(V) for c, V, m, h, n in zip(self.compartments, Vs, ms, hs, ns)]
         
        # Get injected current at each time point
        I_inj_values = np.array([I_inj(ti) for ti in t])
         
        # normalize N(0,1) and add noise
        # noise_std = 1e-4
        Vs_normalized = np.zeros((Vs.shape))
        Vs_obs = np.zeros((Vs.shape))
         
        # Fit the model to the standardized voltage
        # Add a small amount of noise
        for j in range(Vs.shape[1]):
            V = Vs[:,j]
            Vs_normalized[:,j] = (V - V.mean()) / V.std()
            Vs_obs[:,j] = Vs_normalized[:,j] + noise_std * np.random.randn(*V.shape)
         
        # Return the simulated states and currents
        return Vs, ms, hs, ns, I_inj_values, Vs_normalized, Vs_obs

    def plot_simulation(self, t, Vs, ms, hs, ns, I_inj_values):
        C = self.num_compartments
        plt.figure()
        
        for c in range(C):
            plt.subplot(4, C, c + 1)
            plt.title('Hodgkin-Huxley Neuron')
            plt.plot(t, Vs[:,c], 'k')
            plt.ylabel('V (mV)')
        
        for c in range(C):
            plt.subplot(4, C, C + c + 1)
            plt.plot(t, self.compartments[c].I_Na(Vs[:,c], ms[:,c], hs[:,c]), 'c', label='$I_{Na}$')
            plt.plot(t, self.compartments[c].I_K(Vs[:,c], ns[:,c]), 'y', label='$I_{K}$')
            plt.plot(t, self.compartments[c].I_L(Vs[:,c]), 'm', label='$I_{L}$')
            plt.ylabel('Current')
            plt.legend()
        
        for c in range(C):
            plt.subplot(4,C, 2*C + c + 1)
            plt.plot(t, ms[:,c], 'r', label='m')
            plt.plot(t, hs[:,c], 'g', label='h')
            plt.plot(t, ns[:,c], 'b', label='n')
            plt.ylabel('Gating Value')
            plt.legend()
        
        for c in range(C):
            plt.subplot(4, C, 3*C + c + 1)
            plt.plot(t, I_inj_values[:,c], 'k')
            plt.xlabel('t (ms)')
            plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
            plt.ylim(-1, 40)
        
        plt.show()


def make_roslds(V, I_inj_values, not_noisy=True, K=6, D_latent=2, sigmasq_value=1e-4, penalty=.05):
    """
        :param V: the (T,) array of voltage observations
        :param K: the number of discrete states (integer)
        :param D_latent: the dimension of the continuous latent states
        """
    
    assert V.ndim == 1, "V must be a shape (T,) array of voltage observations"
    T = V.shape[0]
    D_obs = 2
    
    '''
    # initialization
    w1, b1 = np.array([+1.0, 0.0]), np.array([0.0])   # x >0
    w2, b2 = np.array([0.0, +1.0]), np.array([0.0])    # y > 0
    
    reg_W = np.row_stack((w1, w2))
    reg_b = np.row_stack((b1, b2))
    '''
    reg_W, reg_b = make_initial_plane(K)
    # Scale the weights to make the transition boundary sharper
    reg_scale = 100.
    reg_b *= reg_scale
    reg_W *= reg_scale
    
    
    # Create the model components
    # (initial state dist, dynamics dist, emissions dist)
    init_dynamics_distns = [Gaussian(mu=np.zeros(D_latent),
                                     sigma=np.eye(D_latent),
                                     nu_0=D_latent + 2, sigma_0=3 * np.eye(D_latent),
                                     mu_0=np.zeros(D_latent), kappa_0=1.0,
                                     ) for _ in range(K)]
        
    dynamics_distns = [Regression( nu_0=D_latent + 4,
                                  S_0=1e-4 * np.eye(D_latent),
                                  M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, 2)))),
                                  K_0=np.eye(D_latent + 2),
                                  affine=False
                                  ) for _ in range(K)]
                            
    # Constrain the emission matrix to have C = [[1, 0, ..., 0]]
    # and small sigmasq
    # C = np.hstack((np.eye(D_obs), np.zeros((D_obs, 2))))
    C = np.hstack((np.eye(D_obs), np.zeros((D_latent, 2))))
    #sigmasq = np.concatenate((np.array([1e-4]), np.ones(D_obs-1)))
    sigmasq = np.array([sigmasq_value, penalty])
    emission_distns = DiagonalRegression(D_obs, D_latent+2, A=C, sigmasq=sigmasq)

    # Construct the full model

    if K == 1:
        rslds = PGRecurrentOnlySLDS(trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.),
                                    init_state_distn='uniform',
                                    init_dynamics_distns=init_dynamics_distns,
                                    dynamics_distns=dynamics_distns,
                                    emission_distns=emission_distns,
                                    fixed_emission=True)
    else:
        rslds = PGRecurrentOnlySLDS(trans_params=dict(A=np.hstack((np.zeros((K-1, K)), reg_W)), b=reg_b, sigmasq_A=10000., sigmasq_b=10000.), init_state_distn='uniform', init_dynamics_distns=init_dynamics_distns, dynamics_distns=dynamics_distns, emission_distns=emission_distns, fixed_emission=True)

    # Initialize the continuous states to be V and its gradient
    assert D_latent == 2


    if not_noisy:
        dV = gaussian_filter1d(np.gradient(V), 1)
    else:
        nconvolve =10000
        dV =np.convolve(np.gradient(V), np.ones((nconvolve,))/nconvolve, mode='same')


    x_init = np.column_stack((V, dV))
    x_init = (x_init - np.mean(x_init, axis=0)) / np.std(x_init, axis=0)

    # Initialize the discrete states by clustering x_init
    km = KMeans(K).fit(x_init)
    z_init = km.labels_


    # Provide an array of ones for the bias (affine term)
    assert I_inj_values.shape == (T,)
    data = np.column_stack((V, np.zeros(T,))) # pseudo obs
    inputs = np.column_stack((np.ones((T, 1)), I_inj_values))
    mask = np.ones((T, D_obs), dtype=bool)
    rslds.add_data(data, mask=mask, inputs=inputs,
                   stateseq=z_init, gaussian_states=x_init)

    return rslds

def run_gibbs_sampler(rslds, N_samples=2000):
    # Resample auxiliary variables
    for states in rslds.states_list:
        states.resample_transition_auxiliary_variables()
    
    # Initialize dynamics
    #print("Initializing dynamics with Gibbs sampling")
    #for itr in tqdm(range(100)):
    for itr in  range(100):
        rslds.resample_dynamics_distns()
    
    # print("Initializing transitions with Gibbs sampling")
    #for itr in tqdm(range(100)):
    for itr in  range(100):
        rslds.resample_trans_distn()
    
    # Fit the model
    #print("Fitting input only rSLDS to the voltage trace")
    lps = []
    z_smpls = [rslds.stateseqs[0].copy()]
    #z_smpls = [rslds.stateseqs.copy()]
    #for itr in tqdm(range(N_samples)):
    for itr in range(N_samples):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())
    #z_smpls.append(rslds.stateseqs.copy())
    
    x_smpl = rslds.states_list[0].gaussian_states
    #x_smpl = rslds.states_list.gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)
    return lps, z_smpls, x_smpl


def load_realdata_ephys(cell_specimen_id=464212183):
    from allensdk.core.cell_types_cache import CellTypesCache
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    
    # this saves the NWB file to 'cell_types/specimen_464212183/ephys.nwb'
    cell_specimen_id =cell_specimen_id#64212183#325941643#464212183 (good)
    data_set = ctc.get_ephys_data(cell_specimen_id)
    
    sweep_number = 35
    sweep_data = data_set.get_sweep(sweep_number)
    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1]+1] # in A
    v = sweep_data["response"][0:index_range[1]+1] # in V
    i *= 1e6 # to mu A #1e12 # to pA
    v *= 1e3 # to mV
    
    sampling_rate = sweep_data["sampling_rate"] # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)
    
    
    D_latent= 2
    signal_idx = np.logical_and(t >= 1.03, t <= 2.022)
    V = v[signal_idx]
    I_inj_values = i[signal_idx]
    t = t[signal_idx]
    t *= 1e3  #ms
    
    
    downsample = 30 # downsample
    V = V[::downsample]
    I_inj_values = I_inj_values[::downsample]
    t = t[::downsample]
    
    return V, t, I_inj_values

def find_branch_paths(compartments_byid, branch_indexes):
    # output pathes between two branch points.
    # compartments_byid: key index; value compartments
    # branch_indexes:
    
    def dfs(id, path_so_far):
        ID, _, children = get_info(compartments_byid[id])
        assert(ID == id)
        if len(path_so_far) > 1 and (ID in branch_indexes or children==[]):
            paths.append(path_so_far[:])
            return
        for i in children:
            path_so_far.append(i)
            dfs(i, path_so_far)
            path_so_far.pop()

    paths = []
    for i in compartments_byid:
        if i in branch_indexes:
            dfs(i, [i])
    return paths

def get_info(cmpt):
    ID = cmpt['id']
    xyz = np.array([cmpt['x'], cmpt['y'], cmpt['z']])
    children = cmpt['children']
    return ID, xyz, children


def load_realdata_morphology(cell_id = 464212183, desample_tree_factor=30):
    
    from allensdk.core.cell_types_cache import CellTypesCache
    # load real data
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    # load 3D morphology data
    morphology = ctc.get_reconstruction(cell_id)
    # extract
    compartments = morphology.compartment_list
    compartments_byid = { c['id']: c for c in compartments}
    
    num_children = np.array([len(c['children'])  for c in compartments])
    branch_indexes = np.where(num_children>1)[0] # branch compartment index
    
    # find all paths (terminate at branches)
    paths = find_branch_paths(compartments_byid, branch_indexes)
    
    # de-sample tree and paths
    select_compartments = [] # index selected
    path_compartments = [] # every connections between selected compartments (not limited to branches)
    path_filtered = [] #per branch, only keep selected one
    for k, path in enumerate(paths):
        n = len(path)
        if n < desample_tree_factor:
            select_compartments.extend([path[0], path[-1]])
            path_compartments.append([path[0], path[-1]])
            path_filtered.append([path[0], path[-1]])
        else:
            use = [path[int(i)] for i in np.linspace(0, n - 1, n//desample_tree_factor+1)]
            path_filtered.append(use)
            select_compartments.extend(use)
            for j in range(len(use) - 1):
                path_compartments.append([use[j], use[j+1]])
    select_compartments = np.unique(select_compartments)

    assert len(np.setdiff1d(branch_indexes, select_compartments)) == 0

    # map old id to new id
    new_id = {select_compartments[i]:i for i in range(len(select_compartments))}
        
    # calculate distance between selected compartments
    from numpy import linalg as LA
    C = len(select_compartments)
    dist = np.full((C, C), np.inf)

    for path in path_compartments:
        ID1, xyz1, _ = get_info(compartments_byid[path[0]])
        ID2, xyz2, _ = get_info(compartments_byid[path[1]])
        dist[new_id[path[0]], new_id[path[1]]] = LA.norm(xyz1 - xyz2, 2)
        dist[new_id[path[1]], new_id[path[0]]] = dist[new_id[path[0]], new_id[path[1]]]

    network = 1/dist
    return network, new_id, compartments_byid, paths, select_compartments, path_compartments, path_filtered, compartments, morphology




def make_roslds_mtpl(V, I_inj_values, V_compartment,  K=3, D_latent=2, sigmasq_value=1e-4, penalty=.05):
    """
        :param V: the (T,) array of voltage observations
        :param K: the number of discrete states (integer)
        :param D_latent: the dimension of the continuous latent states
        """
    assert V.ndim == 1, "V must be a shape (T,) array of voltage observations"
    T = V.shape[0]
    D_obs = 2
    
    directory = './results/'
    # set initial plane
    if K > 1:
        reg_W, reg_b = make_initial_plane(K)
        
        
        # Scale the weights to make the transition boundary sharper
        reg_scale = 100.
        reg_b *= reg_scale
        reg_W *= reg_scale
    
    
    
    # Create the model components
    # (initial state dist, dynamics dist, emissions dist)
    init_dynamics_distns = [
                            Gaussian(
                                     mu=np.zeros(D_latent),
                                     sigma=np.eye(D_latent),
                                     nu_0=D_latent + 2, sigma_0=3 * np.eye(D_latent),
                                     mu_0=np.zeros(D_latent), kappa_0=1.0,
                                     )
                            for _ in range(K)]
        
    dynamics_distns = [
                       Regression(
                                  nu_0=D_latent + 6,#4,
                                  S_0=1e-4 * np.eye(D_latent),
                                  M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, 3)))),#2)))),
                                  K_0=np.eye(D_latent + 3),#2),
                                  affine=False
                                  )
                       for _ in range(K)]
 

    # Constrain the emission matrix to have C = [[1, 0, ..., 0]]
    # and small sigmasq
    # C = np.hstack((np.eye(D_obs), np.zeros((D_obs, 2))))
    C = np.hstack((np.eye(D_obs), np.zeros((D_latent, 3))))#2))))
    #sigmasq = np.concatenate((np.array([1e-4]), np.ones(D_obs-1)))
    sigmasq = np.array([sigmasq_value, penalty])
    emission_distns = \
        DiagonalRegression(D_obs, D_latent+3,#+2,
                           A=C, sigmasq=sigmasq)

    # Construct the full model

    if K == 1:
        rslds = PGRecurrentOnlySLDS(
                                    trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.),
                                    init_state_distn='uniform',
                                    init_dynamics_distns=init_dynamics_distns,
                                    dynamics_distns=dynamics_distns,
                                    emission_distns=emission_distns,
                                    fixed_emission=True)
    else:
        rslds = PGRecurrentOnlySLDS(
                                trans_params=dict(A=np.hstack((np.zeros((K-1, K)), reg_W)), b=reg_b, sigmasq_A=10000., sigmasq_b=10000.),
                                init_state_distn='uniform',
                                init_dynamics_distns=init_dynamics_distns,
                                dynamics_distns=dynamics_distns,
                                emission_distns=emission_distns,
                                fixed_emission=True)

    # Initialize the continuous states to be V and its gradient
    assert D_latent == 2
    from scipy.ndimage import gaussian_filter1d
    from sklearn.cluster import KMeans
    #dV = gaussian_filter1d(np.gradient(V), 1)
    #nconvolve = 0
    #nconvolve =500#100
    #dV =np.convolve(np.gradient(V), np.ones((nconvolve,))/nconvolve, mode='same')
    V_tmp = V.copy()
    v_thre = 0#3.3 (good)#3#3.5#4#3.5#3#2.5#2.2
    V_tmp[V_tmp<v_thre] = 0
    print('NEW INITIALIZATION!', v_thre)
    #dV = gaussian_filter1d(np.gradient(V_tmp), 1)
    dV = gaussian_filter1d(np.gradient(V_tmp), 10)
    print('convolue dV')
    #x_init = np.column_stack((V, dV))
    x_init = np.column_stack((V_tmp, dV))
    x_init = (x_init - np.mean(x_init, axis=0)) / np.std(x_init, axis=0)
    
    # Initialize the discrete states by clustering x_init
    km = KMeans(K).fit(x_init)
    z_init = km.labels_
    
    # Plot the
    #'''
    plt.close('all')
    plt.figure()
    plt.subplot(211)
    plt.plot(x_init[:10000,:])
    # plt.plot(Vs_true[:,j],'r')
    plt.subplot(212)
    plt.plot(z_init[:10000])
    
    #plt.show()
    plt.savefig(directory+'-init-penalized.png')
    #plt.savefig('nconvolve'+str(nconvolve)+'-init-penalized.png')
    #'''
    
    # Provide an array of ones for the bias (affine term)
    assert I_inj_values.shape == (T,)
    data = np.column_stack((V, np.zeros(T,))) # pseudo obs
    # add voltage effect on rslds (V_compartments)
    inputs = np.column_stack((np.ones((T, 1)), I_inj_values, V_compartment))
    #inputs = np.column_stack((np.ones((T, 1)), I_inj_values))
    mask = np.ones((T, D_obs), dtype=bool)
    rslds.add_data(data, mask=mask, inputs=inputs,
                   stateseq=z_init, gaussian_states=x_init)
                   
    return rslds

def make_initial_plane(K, grid_sz=2.0, xshift=0.0, yshift=0.0):
    # grid_sz: size of grid
    # xshift: positive: x moves right; negative: x moves to left
    # yshift: same as x
    if K == 1:
        return
    
    if K == 2:
        w1, b1 = np.array([+1.0, 0.0]), np.array([0.0])
        reg_W = np.row_stack((w1))
        reg_b = np.row_stack((b1))
    
    if K == 4:
        w1, b1 = np.array([+1.0, 0.0]), np.array([0.0])    # if x > 0: (0, 1) else (2, 3)
        w2, b2 = np.array([0.0, +1.0]), np.array([0.0])    # if y > 0 and in (0, 1): 0 else 1
        w3, b3 = np.array([0.0, +1.0]), np.array([0.0])    #
        reg_W = np.row_stack((w1, w2, w3))
        reg_b = np.row_stack((b1, b2, b3))
    if K == 3:
        # initialization
        w1, b1 = np.array([+1.0, 0.0]), np.array([0.0])   # x >0
        w2, b2 = np.array([0.0, +1.0]), np.array([0.0])    # y > 0
        
        reg_W = np.row_stack((w1, w2))
        reg_b = np.row_stack((b1, b2))
        
        reg_W = np.row_stack((w1, w2))
        reg_b = np.row_stack((b1, b2))
    
    
    if K == 8:
        w1, b1 = np.array([+1.0, 0.0]), np.array([0.0 - xshift])    # if x > 0: (0, 1) else (2, 3)
        w2, b2 = np.array([0.0, +1.0]), np.array([0.0 - yshift])    # if y > 0 and in (0, 1): 0 else 1
        w3, b3 = np.array([+1.0, 0.0]), np.array([-grid_sz - xshift])   #
        w4, b4 = np.array([+1.0, 0.0]), np.array([-grid_sz - xshift])   #
        w5, b5 = np.array([0.0, +1.0]), np.array([0.0 - yshift])    # if y > 0 and in (2, 3): 2 else 3
        w6, b6 = np.array([+1.0, 0.0]), np.array([+grid_sz - xshift])   #
        w7, b7 = np.array([+1.0, 0.0]), np.array([+grid_sz - xshift])   #
        reg_W = np.row_stack((w1, w2, w3, w4, w5, w6, w7))
        reg_b = np.row_stack((b1, b2, b3, b4, b5, b6, b7))
    
    if K == 16:
        
        w1, b1 = np.array([+1.0, 0.0]), np.array([0.0 - xshift])    # if x > 0: (0, 1) else (2, 3)
        w2, b2 = np.array([0.0, +1.0]), np.array([0.0 - yshift])    # if y > 0 and in (0, 1): 0 else 1
        w3, b3 = np.array([+1.0, 0.0]), np.array([-grid_sz - xshift])   #
        w4, b4 = np.array([0, 1.0]), np.array([-grid_sz - yshift])   #
        w5, b5 = np.array([0, 1.0]), np.array([-grid_sz - yshift])   #
        
        w6, b6 = np.array([+1.0, 0.0]), np.array([-grid_sz - xshift])   #
        w7, b7 = np.array([0, 1.0]), np.array([grid_sz - yshift])
        w8, b8 = np.array([0, 1.0]), np.array([grid_sz - yshift])
        
        w9, b9 = np.array([0.0, +1.0]), np.array([0.0 - yshift])    # if y > 0 and in (2, 3): 2 else 3
        w10, b10 = np.array([+1.0, 0.0]), np.array([+grid_sz - xshift])   #
        w11, b11 = np.array([0, 1.0]), np.array([-grid_sz - yshift])   #
        w12, b12 = np.array([0, 1.0]), np.array([-grid_sz - yshift])   #
        
        w13, b13 = np.array([+1.0, 0.0]), np.array([+grid_sz - xshift])   #
        w14, b14 = np.array([0, 1.0]), np.array([grid_sz - yshift])   #
        w15, b15 = np.array([0, 1.0]), np.array([grid_sz - yshift])   #
        
        
        reg_W = np.row_stack((w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15))
        reg_b = np.row_stack((b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 ))
    
    return reg_W, reg_b










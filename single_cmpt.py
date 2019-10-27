import numpy as np
import numpy.random as npr
from sklearn.metrics import mean_squared_error
from NIPS_utils import load_realdata_ephys, make_roslds, run_gibbs_sampler

def run_real_data(K=3, noise_std=.2, sigmasq_value=.05, penalty=.1):

    seed = 0
    npr.seed(seed)
    
    V, t, I_inj_values = load_realdata_ephys(cell_specimen_id=464212183)
    
    N_samples = 1000
    D_latent= 2
    
    # Fit the model to the standardized voltage
    V_true = (V - V.mean()) / V.std()
    
    # Add a small amount of noise
    V_obs = V_true + noise_std * np.random.randn(*V.shape)
    
    # Construct the recurrent SLDS
    rslds = make_roslds(V_obs, I_inj_values, K=K, D_latent=D_latent, sigmasq_value=sigmasq_value, penalty=penalty)
    # run recurrent SLDS
    lps, z_smpls, x_smpl, x_smpls = run_gibbs_sampler(rslds, N_samples=N_samples)
    mse = mean_squared_error(V_true, x_smpl[:,0])

    #return mse, x_smpl, V_true, V_obs, rslds_x_gen_hist, rslds_z_gen_hist, rslds, z_smpls, lps, x_smpls, t
    return mse, x_smpl, V_true, V_obs, rslds, z_smpls, lps, x_smpls, t, I_inj_values






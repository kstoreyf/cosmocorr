import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM


zmin = 0.42
zmax = 0.71
precision = 0.0001
zs = np.arange(zmin, zmax, precision)

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

def get_dcm(z):
    return cosmo.comoving_distance(z).value

def get_transverse_dcm(z):
    return cosmo.comoving_transverse_distance(z).value

df = pd.DataFrame(zs, columns=['z_round'])

df['dcm_mpc'] = df['z_round'].apply(get_dcm)
df['dcm_transverse_mpc'] = df['z_round'].apply(get_transverse_dcm)

df.to_csv('data/z_lookup_lcdm_H070_Om0.3_Ode0.7_4dec.csv', index=False)
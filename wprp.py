import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from astropy.cosmology import LambdaCDM
import time
from scipy import interpolate


def wprp(ra_data, dec_data, z_data, ra_rand, dec_rand, z_rand, rpbins, pimax,
         weights_data=None, weights_rand=None, pibinwidth=1, comoving=False, zfile=None):

    assert len(ra_data)==len(dec_data) and len(ra_data)==len(z_data), "Data arrays have different lengths"
    assert len(ra_rand)==len(dec_rand) and len(ra_rand)==len(z_rand), "Random arrays have different lengths"

    ndata = len(ra_data)
    nrand = len(ra_rand)

    if weights_data==None:
        weights_data = np.ones(ndata)
    if weights_rand==None:
        weights_rand = np.ones(nrand)
    if zfile==None:
        zfile = 'data/z_lookup_lcdm_H070_Om0.3_Ode0.7_4dec.csv'

    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    zdf = pd.read_csv(zfile)

    interp_dcm = interpolate.interp1d(zdf['z_round'], zdf['dcm_mpc'])
    interp_dcm_transverse = interpolate.interp1d(zdf['z_round'], zdf['dcm_transverse_mpc'])

    print 'Building dataframes and applying spherical projections'
    datadf = pd.DataFrame(np.array([ra_data, dec_data, z_data, weights_data]).T, columns=['ra', 'dec', 'z', 'w'])
    datadf['xproj'], datadf['yproj'], datadf['zproj'] = zip(*datadf.apply(ra_dec_to_unitxyz, axis=1))
    randdf = pd.DataFrame(np.array([ra_rand, dec_rand, z_rand, weights_rand]).T, columns=['ra', 'dec', 'z', 'w'])
    randdf['xproj'], randdf['yproj'], randdf['zproj'] = zip(*randdf.apply(ra_dec_to_unitxyz, axis=1))

    datadf['dcm_mpc'] = datadf['z'].apply(interp_dcm)
    datadf['dcm_transverse_mpc'] = datadf['z'].apply(interp_dcm_transverse)

    randdf['dcm_mpc'] = randdf['z'].apply(interp_dcm)
    randdf['dcm_transverse_mpc'] = randdf['z'].apply(interp_dcm_transverse)

    print 'Building data KDTree...'
    start = time.time()
    datatree = KDTree(list(np.array([datadf['xproj'], datadf['yproj'], datadf['zproj']]).T))
    end = time.time()
    print 'Constructed data tree: {:.4f}'.format(end-start)
    print 'Building random KDTree...'
    start = time.time()
    randtree = KDTree(list(np.array([randdf['xproj'], randdf['yproj'], randdf['zproj']]).T))
    end = time.time()
    print 'Constructed random tree: {:.4f}'.format(end-start)

    print 'Counting DD pairs...'
    start = time.time()
    dd_rp_pi = count_pairs(datadf, datadf, datatree, rpbins, pimax, pibinwidth, cosmo)
    end = time.time()
    print 'DD counts: {:.4f}'.format(end-start)
    print 'Counting DR pairs...'
    start = time.time()
    dr_rp_pi = count_pairs(randdf, datadf, datatree, rpbins, pimax, pibinwidth, cosmo)
    end = time.time()
    print 'DR counts: {:.4f}'.format(end-start)
    print 'Counting RR pairs...'
    start = time.time()
    rr_rp_pi = count_pairs(randdf, randdf, randtree, rpbins, pimax, pibinwidth, cosmo)
    end = time.time()
    print 'RR counts: {:.4f}'.format(end-start)
    #print dd_rp_pi
    #print dd_rp_pi
    #print rr_rp_pi
    estimator = calc_estimator(dd_rp_pi, dr_rp_pi, rr_rp_pi, ndata, nrand)
    wprp = 2*pibinwidth*np.sum(estimator, axis=0)
    return estimator, wprp


def count_pairs(df_cat, df_tree, tree, rpbins, pimax, pibinwidth, cosmo):

    pibins = np.arange(0, pimax + pibinwidth, pibinwidth)
    counts_rp_pi = np.zeros((len(pibins) - 1, len(rpbins) - 1))

    ncat = len(df_cat)
    ntree = len(df_tree)
    print ncat, ntree

    xproj_cat = df_cat['xproj'].values
    yproj_cat = df_cat['yproj'].values
    zproj_cat = df_cat['zproj'].values
    w_cat = df_cat['w'].values
    w_tree = df_tree['w'].values
    dcm_cat = df_cat['dcm_mpc'].values
    dcm_tree = df_tree['dcm_mpc'].values
    dcm_transverse_cat = df_cat['dcm_transverse_mpc'].values
    h = cosmo.h

    for i in range(ncat):

        xproj = xproj_cat[i]
        yproj = yproj_cat[i]
        zproj = zproj_cat[i]
        rmax_unit = rpbins[-1]/(dcm_transverse_cat[i]*h)
        dists_max, locs_max = tree.query([xproj, yproj, zproj], k=ntree, distance_upper_bound=rmax_unit)

        # Query returns infinities when <k neighbors are found, cut these
        if locs_max[-1] == ntree:
            imax = next(index for index, value in enumerate(locs_max) if value == ntree)
            locs_max = locs_max[:imax]

        unit_bins = [rpb/(dcm_transverse_cat[i]*h) for rpb in rpbins]
        pos = np.digitize(dists_max, unit_bins)

        for k in range(len(locs_max)):

            loc = locs_max[k]
            irp = pos[k]-1
            pi = h*abs(dcm_cat[i] - dcm_tree[loc])
            if pi < pimax and irp>=0 and irp<len(rpbins)-1:
                ipi = int(np.floor(pi/pibinwidth))
                counts_rp_pi[ipi][irp] += (w_cat[i]+w_tree[loc])/2.0

        if i%100==0:
           print i

    return counts_rp_pi


# Ra, dec in degrees
def ra_dec_to_unitxyz(row):
    ra = row['ra'] * np.pi / 180
    dec = row['dec'] * np.pi / 180
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    return x, y, z

def calc_estimator(dd_counts, dr_counts, rr_counts, ndata, nrand, estimator='ls'):
    fN = float(nrand)/float(ndata)
    if estimator=='ls':
        return (fN*fN*dd_counts - 2*fN*dr_counts + rr_counts)/rr_counts
    else:
        raise ValueError('Estimator {} not supported, try ls.'.format(estimator))

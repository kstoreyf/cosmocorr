import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import time

import wprp_corrfunc
import wprp


#globals
plot_dir = 'plots_8-24/'
rpbins = np.logspace(np.log10(0.2), np.log10(20), 10)
pimax = 20
colors = ['blue', 'magenta', 'orange', 'green', 'red', 'purple']


def main():
    print 'Reading data...'

    nd = 102
    datadf = pd.read_csv('../mangler/samples/a0.6452_0001.v5_ngc_ifield_ndata{}.rdzw'.format(nd))
    randdf = pd.read_csv('../mangler/samples/a0.6452_rand20x.dr12d_cmass_ngc_ifield_ndata{}.rdz'.format(nd))

    print 'n_data: {}, n_rand: {}'.format(len(datadf), len(randdf))

    print 'pi max:', pimax
    print 'rp bins:', rpbins


    start = time.time()
    estimator, wp = wprp.wprp(datadf['ra'].values, datadf['dec'].values, datadf['z'].values,
        randdf['ra'].values, randdf['dec'].values, randdf['z'].values, rpbins, pimax)
    end = time.time()
    print 'Total time: {:.4f}'.format(end-start)
    print wp

    start = time.time()
    estimator_corrfunc_comoving, wp_corrfunc_comoving = wprp_corrfunc.wprp(
        datadf['ra'].values, datadf['dec'].values, datadf['z'].values,
        randdf['ra'].values, randdf['dec'].values, randdf['z'].values,
        rpbins, pimax, comoving=True)
    end = time.time()
    print 'Time:', end-start
    print wp_corrfunc_comoving

    plot_wprp([wp], ['cosmocorr'], wp_corrfunc_comoving, 'Corrfunc comoving')

    plt.show()


def mirror_array(arr):
    top = np.concatenate((np.flip(np.flip(arr, 0), 1), np.flip(arr, 0)), axis=1)
    bot = np.concatenate((np.flip(arr, 1), arr), axis=1)
    return np.concatenate((top, bot), axis=0)

def plot_pi_rp(estimator, saveto=None):
    fig = plt.figure()
    im = plt.imshow(estimator, norm=LogNorm(), cmap='plasma', interpolation=None, origin='lower', extent=[min(rpbins), max(rpbins), 0, pimax])
    plt.xlabel(r'r$_p$ (Mpc/h)')
    plt.ylabel(r'$\pi$ (Mpc/h)')
    fig.colorbar(im)
    if saveto:
        plt.savefig(plot_dir+saveto)

def plot_pi_rp_mirror(estimator, saveto=None):
    fig = plt.figure()
    im = plt.imshow(mirror_array(estimator), norm=LogNorm(), cmap='plasma', interpolation=None, origin='lower', extent=[-1*max(rpbins), max(rpbins), - 1*pimax, pimax])
    plt.xlabel(r'r$_p$ (Mpc/h)')
    plt.ylabel(r'$\pi$ (Mpc/h)')
    fig.colorbar(im)
    if saveto:
        plt.savefig(plot_dir+saveto)

def plot_wprp(wps, labels, wp_tocompare, label_tocompare, saveto=None):
    wps = [wp_tocompare] + wps
    labels = [label_tocompare] + labels
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    for i in range(len(wps)):
        wp = wps[i]
        label = labels[i]
        rpbins_avg = [(rpbins[j]+rpbins[j+1])/2 for j in range(len(rpbins)-1)]
        ax0.loglog(rpbins_avg, wp/rpbins_avg, color=colors[i], marker='o', label=label)

        ax0.legend(loc='best')
        plt.xlabel(r'r$_p$ (Mpc/h)')

        ax1.set_ylabel(r'error')
        ax0.set_ylabel(r'w$_p$(r$_p$)/r$_p$')

        ax1.semilogx(rpbins_avg, wp/wp_tocompare, color=colors[i])

    if saveto:
        plt.savefig(plot_dir+saveto)

def plot_distributions(datadf, randdf, saveto=None):
    fig, axarr = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    w_d = np.ones_like(datadf['ra'])/float(len(datadf['ra']))
    w_r = np.ones_like(randdf['ra'])/float(len(randdf['ra']))
    axarr[0].hist(datadf['ra'], bins=15, histtype='step', weights=w_d)
    axarr[0].hist(randdf['ra'], bins=15, histtype='step', weights=w_r)
    axarr[0].set_xlabel('ra')
    axarr[0].set_ylabel('%')
    axarr[1].hist(datadf['dec'], bins=15, histtype='step', weights=w_d)
    axarr[1].hist(randdf['dec'], bins=15, histtype='step', weights=w_r)
    axarr[1].set_xlabel('dec')
    axarr[2].hist(datadf['z'], bins=15, histtype='step', weights=w_d, label='data')
    axarr[2].hist(randdf['z'], bins=15, histtype='step', weights=w_r, label='randoms')
    axarr[2].set_xlabel('z')
    axarr[2].legend(loc='best')
    if saveto:
        plt.savefig(plot_dir+saveto)


if __name__=='__main__':
    main()
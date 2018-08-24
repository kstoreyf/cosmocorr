import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

import wprp_corrfunc
import wprp


#globals
ndata = [10, 31, 102, 307, 1012, 3158]
cols = ['blue', 'magenta', 'orange', 'green', 'red', 'purple', 'brown']
plotdir = 'plots_8-24/'


def main():

    times_corrfunc = np.zeros(len(ndata))
    times_cosmocorr = np.zeros(len(ndata))

    rpbins = np.logspace(np.log10(0.2), np.log10(20), 10)
    pimax = 20
    print 'pi max:', pimax
    print 'rp bins:', rpbins

    for i in range(len(ndata)):

        nd = ndata[i]

        datadf = pd.read_csv('../mangler/samples/a0.6452_0001.v5_ngc_ifield_ndata{}.rdzw'.format(nd))
        randdf = pd.read_csv('../mangler/samples/a0.6452_rand20x.dr12d_cmass_ngc_ifield_ndata{}.rdz'.format(nd))

        print 'n_data: {}, n_rand: {}'.format(len(datadf), len(randdf))

        start0 = time.time()
        wprp_corrfunc.wprp(
             np.copy(datadf['ra'].values), np.copy(datadf['dec'].values),
             np.copy(datadf['z'].values), np.copy(randdf['ra'].values),
             np.copy(randdf['dec'].values), np.copy(randdf['z'].values), rpbins, pimax, comoving=False)
        end0 = time.time()
        times_corrfunc[i] = end0 - start0


        start1 = time.time()
        wprp.wprp(np.copy(datadf['ra'].values), np.copy(datadf['dec'].values),
                                           np.copy(datadf['z'].values), np.copy(randdf['ra'].values),
                                           np.copy(randdf['dec'].values), np.copy(randdf['z'].values), rpbins, pimax)
        end1= time.time()
        times_cosmocorr[i] = end1 - start1

        time_arrs = [times_corrfunc, times_cosmocorr]

        plot_times(time_arrs, nd)



def plot_times(time_arrs, nd):

    plt.figure()
    print 'Plotting', nd

    times_corrfunc, times_cosmocorrv2, times_cosmocorrv3, times_cosmocorrv4, times_cosmocorrv5, times_qpv3 = time_arrs

    logndata = np.log10(ndata)

    fit0 = np.polyfit(logndata, np.log10(times_corrfunc), 1)
    yy = np.array(logndata)*fit0[0]+fit0[1]
    plt.plot(logndata, yy, ls='--', color=cols[0])

    fit1 = np.polyfit(logndata, np.log10(times_cosmocorrv1), 1)
    yy = np.array(logndata)*fit1[0]+fit1[1]
    plt.plot(logndata, yy, ls='--', color=cols[1])

    plt.plot(logndata, np.log10(times_corrfunc), marker='o', label='corrfunc: m={:.2f}'.format(fit0[0]), ls='-', color=cols[0])
    plt.plot(logndata, np.log10(times_cosmocorr), marker='o', label='cosmocorr: m={:.2f}'.format(fit1[0]), ls='-', color=cols[1])

    plt.legend(loc='best')

    plt.xlabel(r'log(n$_{data}$)')
    plt.ylabel('log(seconds)')

    plotname = plotdir+'time_ndata{}_dense_v345_again.png'.format(nd)
    print 'Saving to', plotname
    plt.savefig(plotname)


if __name__=='__main__':
    main()
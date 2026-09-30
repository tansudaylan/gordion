from tdpy.verbosity import print
import numpy as np

import time as timemodu

import pickle

import scipy.signal
from scipy import interpolate

import os, sys, datetime, fnmatch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astroquery.mast import Catalogs
import astroquery

import astropy
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
import astropy.time

import multiprocessing

try:
    import tcat.main as tcatmain
except ModuleNotFoundError:  # pragma: no cover - optional dependency for some workflows
    tcatmain = None

try:
    from lion import main as lionmain
except ModuleNotFoundError:  # pragma: no cover - optional dependency for some workflows
    lionmain = None

import tdpy
from tdpy.util import summgene


def require_optional_dependency(module_name, dep_name):
    if module_name is None:
        raise ModuleNotFoundError(
            f"{dep_name} is required for this gordion workflow but is not installed in the current environment."
        )


def retr_timeexec():
    # input PCAT speed per 100x100 pixel region
    timeregi = 30. # [min]
    
    # number of time frames in each region
    numbtser = 13.7 * 4 * 24 * 60. / 30.
    
    timeregitser = numbtser * timeregi / 60. / 24 # [day]
    timeffim = 16.8e6 / 1e4 * timeregi # [day]
    timesegm = 4. * timeffim / 7. # [week]
    timefsky = 26 * timesegm / 7. # [week]
    

def plot_peri(): 
    
    ## plot Lomb Scargle periodogram
    figr, axis = plt.subplots(figsize=(12, 4))
    axis.set_ylabel('Power')
    axis.set_xlabel('Frequency [1/day]')
    arryfreq = np.linspace(0.1, 10., 2000)
    for a in range(2):
        indxtemp = np.arange(arryseco.shape[0])
        if a == 0:
            colr = 'g'
        if a == 1:
            colr = 'r'
            for k in range(1400, 1500):
                indxtemp = np.setdiff1d(indxtemp, np.where(abs(arryseco[:, 0] - k * peri - epoc) < dura * 2)[0])
        ydat = scipy.signal.lombscargle(arryseco[indxtemp, 0], arryseco[indxtemp, 1], arryfreq)
        axis.plot(arryfreq * 2. * np.pi, ydat, ls='', marker='o', markersize=5, alpha=0.3, color=colr)
    for a in range(4):
        axis.axvline(a / peri, ls='--', color='black')
    plt.tight_layout()
    path = pathimag + 'lspd_%s.%s' % (strgmask, gdat.strgplotextn)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()


def plot_imag(gdat, cntp, strgfileinit, strgfilefinl='', path=None, cmap=None, indxsideyposoffs=0, indxsidexposoffs=0, boolcatl=False, booloplt=False, \
                         strgtitlinit='', strgtitlextn='', boolresi=False, xposoffs=None, yposoffs=None, indxpixlcolr=None, vmin=None, vmax=None):
    
    if cmap == None:
        if boolresi:
            cmap = 'RdBu'
        else:
            cmap = 'Greys_r'
    
    if vmin is None or vmax is None:
        vmax = np.amax(cntp)
        vmin = np.amin(cntp)
        if boolresi:
            vmax = max(abs(vmax), abs(vmin))
    
    if gdat.cntpscaltype == 'asnh':
        cntp = np.arcsinh(cntp)
        vmin = np.arcsinh(vmin)
        vmax = np.arcsinh(vmax)

    figr, axis = plt.subplots(figsize=(8, 6))
    objtimag = axis.imshow(cntp, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
    if indxpixlcolr is not None:
        temp = np.zeros_like(cntp).flatten()
        temp[indxpixlcolr[-1]] = 1.
        temp = temp.reshape((gdat.numbside, gdat.numbside))
        alph = np.zeros_like(cntp).flatten()
        alph[indxpixlcolr[-1]] = 1.
        alph = alph.reshape((gdat.numbside, gdat.numbside))
        alph = np.copy(temp)
        axis.imshow(temp, origin='lower', interpolation='nearest', alpha=0.5)
    
    axis.set_title(strgtitlinit + gdat.strgtitl + strgtitlextn)

    # overplot catalog
    #plot_catl(gdat, axis, indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs)
    
    # make color bar
    #cax = figr.add_axes([0.83, 0.1, 0.03, 0.8])
    #cbar = figr.colorbar(objtimag, cax=cax)
    #if gdat.cntpscaltype == 'asnh':
    #    tick = cbar.get_ticks()
    #    tick = np.sinh(tick)
    #    labl = ['%d' % int(tick[k]) for k in range(len(tick))]
    #    cbar.set_ticklabels(labl)
    
    if boolcatl:
        axis.scatter(gdat.indxyposposisort, gdat.indxxposposisort, color='b', alpha=0.5, marker='*')
        # temp
        #for n in gdat.indxpixlposi:
        #    axis.text(gdat.indxyposposisort[n], gdat.indxxposposisort[n], '%d' % gdat.g2idposipnts[n])
    if path is None:
        path = gdat.pathimag + '%s_%s%s.%s' % (strgfileinit, gdat.strgcntp, strgfilefinl, gdat.strgplotextn)
    print('Writing to %s...' % path)
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


plot_catl = tdpy.plot_catl


def plot_anim(gdat, maps, strgfileinit, strgtitlinit='', strgtitlextn='', boolcatl=False, booloplt=False):
    
    # plot data
    ## make frames
    pathplotbase = gdat.pathimag + '%s_%s' % (strgfileinit, gdat.strgcntp)
    pathplotanim = '%s.gif' % pathplotbase
    
    if True or os.path.exists(pathplotanim):
        print('Plotting data...')
        listpath = []
        
        vmin = np.amin(maps)
        vmax = np.amax(maps)

        for t in gdat.indxtime:
            print('t')
            print(t)
            strgfilefinl = '_%04d' % t
            path = plot_imag(gdat, maps[:, :, t], strgfileinit, booloplt=booloplt, boolcatl=boolcatl, strgfilefinl=strgfilefinl)
            listpath.append(path)
            
    ## make animation
    print('Making an animation...')
    cmnd = 'convert -delay 10 '
    for path in listpath:
        cmnd += '%s ' % path
    cmnd += pathplotanim
    os.system(cmnd)
    for path in listpath:
        os.system('rm -rf %s' % path)


def main( \
         listisec=None, \
         listicam=None, \
         listiccd=None, \
         datatype='obsd', \
         
         strgmode='full', \

         rasctarg=None, \
         decltarg=None, \
         labltarg=None, \
         strgtarg=None, \
         numbside=None, \
         
         boolrebn=False, \

         **args \
        ):
    
    # preliminary setup
    # construct the global object 
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().iteritems():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # copy all provided inputs to the global object
    for strg, valu in args.iteritems():
        setattr(gdat, strg, valu)

    print('gdat.datatype')
    print(gdat.datatype)

    gdat.boolcatl = True
    gdat.strgplotextn = 'png'

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    np.random.seed(45)

    print('TPET initialized at %s...' % gdat.strgtimestmp)
    
    # paths
    ## read TPET path environment variable
    gdat.pathbase = tdpy.retr_pathbase('tpet')
    gdat.pathdata = tdpy.ensr_path(os.path.join(gdat.pathbase, 'data'))
    gdat.pathimag = tdpy.ensr_path(os.path.join(gdat.pathbase, 'imag'))

    ## define paths
    #gdat.pathdataorig = '/pdo/qlp-data/orbit-%d/ffi/cam%d/ccd%d/FITS/' % (isec, icam, iccd)
    gdat.pathdataorig = tdpy.ensr_path(os.path.join(gdat.pathdata, 'ffis'))
    gdat.pathdatainit = tdpy.ensr_path(os.path.join(gdat.pathdata, 'init'))
    gdat.pathdatainitimag = tdpy.ensr_path(os.path.join(gdat.pathdatainit, 'imag'))
    gdat.pathdatainitanim = tdpy.ensr_path(os.path.join(gdat.pathdatainit, 'anim'))
    gdat.pathdatacomm = tdpy.ensr_path(os.path.join(gdat.pathdata, 'comm'))
    
    # settings
    ## user interaction
    verbtype = 1
    
    ## plotting
    gdat.cntpscaltype = 'asnh'
        
    print('Cutout input data are not provided.')
    if gdat.strgmode == 'full':
        numbsideyposfull = 2078
        numbsidexposfull = 2136
        numbpixloffsypos = 30
        numbpixloffsxpos = 44
        gdat.numbside = numbsideyposfull - numbpixloffsypos
        print('numbsidexposfull')
        print(numbsidexposfull)
    else:
        numbpixloffsypos = 0
        numbpixloffsxpos = 0

    if gdat.numbside is None:
        gdat.numbside = 11
    print('gdat.strgmode')
    print(gdat.strgmode)
        
    timeexpo = 1426.
    
    gdat.numbtimerebn = 30
    
    ## parameters
    numbsidecorr = 1
    numbneigaper = 1
    numbstrd = 1
    
    gdat.distsqrd = 4.
    
    gdat.boolquatplot = True
    gdat.boolquat = False

    gdat.offscorr = numbsidecorr / 2

    gdat.numbneigback = 8
    
    gdat.numbsideedge = gdat.numbneigback + gdat.offscorr
    numbtile = (gdat.numbside - gdat.numbsideedge) // 256 + 1
    numbsidesrch = (gdat.numbside - gdat.numbsideedge) / numbtile
    numbsideshft = numbsidesrch + gdat.numbsideedge
    numbsidetile = numbsidesrch + 2 * gdat.numbneigback + 2 * gdat.offscorr
    
    indxtile = np.arange(numbtile)
    indxsidetile = np.arange(numbsidetile)
    
    if gdat.strgmode == 'tcut':
        from astroquery.mast import Tesscut
        from astropy.coordinates import SkyCoord
        cutout_coord = SkyCoord(rasctarg, decltarg, unit="deg")
        listhdundata = Tesscut.get_cutouts(cutout_coord, numbsidetile)
        sector_table = Tesscut.get_sectors(SkyCoord(gdat.rasctarg, gdat.decltarg, unit="deg"))
        listisec = sector_table['sector'].data
        listicam = sector_table['camera'].data
        listiccd = sector_table['ccd'].data
    
    numbsect = len(listisec)
    indxsect = np.arange(numbsect)
    print('numbsect')
    print(numbsect)
    print('listisec')
    print(listisec)
    if gdat.strgmode == 'tcut':
        listobjtwcss = [[] for o in indxsect]
        if len(listhdundata) == 0:
            raise Exception('TESSCut could not find any data.')
    
    print('numbsidetile')
    print(numbsidetile)

    # number of positives
    gdat.numbpixlposi = 10
    gdat.indxpixlposi = np.arange(gdat.numbpixlposi)
    
    
    for o in indxsect:

        # check inputs
        np.set_printoptions(precision=3, linewidth=180)
        print('Sector: %d' % listisec[o])
        print('Camera: %d' % listicam[o])
        print('CCD: %d' % listiccd[o])
        
        strgsecc = '%02d%d%d' % (listisec[o], listicam[o], listiccd[o])

        isec = listisec[o]
        icam = listicam[o]
        iccd = listiccd[o]

        # fix the seed
        if gdat.datatype == 'mock':
            numbsour = 100
            numbsupn = 10
            gdat.numbtime = 100
        else:
            if gdat.strgmode == 'full':
                # get list of paths where FFIs live
                listrtag = fnmatch.filter(os.listdir(gdat.pathdataorig), 'tess*-s%04d-%d-%d-*-s_ffic.fits' % (isec, icam, iccd))
                listrtag.sort()
                print('Found %d files...' % len(listrtag))
                listrtag = listrtag[::30]
                print('Using %d of these...' % len(listrtag))
                gdat.numbtime = len(listrtag)
            if gdat.strgmode == 'tcut':
                listhdundatatemp = listhdundata[o]
                gdat.time = listhdundatatemp[1].data['TIME'].astype(float)
                gdat.time = gdat.time[::30]
                gdat.numbtime = gdat.time.size
                listobjtwcss[o] = WCS(listhdundata[o][2].header)
        print('gdat.numbtime')
        print(gdat.numbtime)
        gdat.indxtime = np.arange(gdat.numbtime)
       
        xpospixl = np.linspace(0.5, numbsidetile - 0.5, numbsidetile)
        ypospixl = np.linspace(0.5, numbsidetile - 0.5, numbsidetile)
        xpospixlmesh, ypospixlmesh = np.meshgrid(xpospixl, ypospixl)
        gdat.indxsidexpos, gdat.indxsideypos = np.meshgrid(indxsidetile, indxsidetile)

        if (gdat.numbside - gdat.numbsideedge) % numbtile != 0:
            raise Exception('')

        numbdata = numbsidesrch**2
        indxdata = np.arange(numbdata)
        
        indxsidesrch = np.arange(numbsidesrch)
        numbsrch = numbsidesrch**2
        
        indxsideypostile, indxsidexpostile = np.meshgrid(indxsidetile, indxsidetile)
        indxsideyposdata = indxsideypostile[gdat.numbsideedge:-gdat.numbsideedge, gdat.numbsideedge:-gdat.numbsideedge] - gdat.numbsideedge
        indxsidexposdata = indxsidexpostile[gdat.numbsideedge:-gdat.numbsideedge, gdat.numbsideedge:-gdat.numbsideedge] - gdat.numbsideedge
        
        if numbdata != indxsideyposdata.size:
            raise Exception('')

        indxsideyposdatatemp = np.empty((numbsidesrch, numbsidesrch, numbsidecorr, numbsidecorr))
        indxsidexposdatatemp = np.empty((numbsidesrch, numbsidesrch, numbsidecorr, numbsidecorr))
        indxsideyposdatatemp[:, :, :, :] = indxsideyposdata[:, :, None, None]
        indxsidexposdatatemp[:, :, :, :] = indxsidexposdata[:, :, None, None]
        gdat.indxsideyposdataflat = indxsideyposdatatemp.flatten().astype(int)
        gdat.indxsidexposdataflat = indxsidexposdatatemp.flatten().astype(int)
        numbpixltile = numbsidetile**2
                
        if gdat.datatype == 'mock':
            listlabltrue = np.zeros(numbdata, dtype=int)
            numbinli = numbdata - numbsour
            numboutl = numbsour
        
        thrsrmss = 0.01
        thrsmaxm = 1.5
        thrsdiff = 0.5
        numbsideaper = 2 * numbneigaper + 1
        numbpixlaper = numbsideaper**2
        
        if float(gdat.numbside) % numbstrd != 0:
            raise Exception('')

        # grid of flux space
        minmproj = 0.1
        maxmproj = 2
        limtproj = [minmproj, maxmproj]
        arry = np.linspace(minmproj, maxmproj, 100)
        xx, yy = np.meshgrid(arry, arry)
        
        magtminm = 12.
        magtmaxm = 19.
        indxsidetilesrch = np.arange(gdat.numbneigback + gdat.offscorr, numbsidesrch + gdat.numbsideedge)
        
        liststrgpara = ['rmss', 'maxm', 'mean', 'diff']
        listlablpara = ['R', 'Max', 'Mean', 'Difference']
        dictpara = {}
        dictpara['rmss'] = np.zeros((numbsidetile, numbsidetile)) - 1.
        dictpara['maxm'] = np.zeros((numbsidetile, numbsidetile)) - 1.
        dictpara['mean'] = np.zeros((numbsidetile, numbsidetile)) - 1.
        dictpara['diff'] = np.zeros((numbsidetile, numbsidetile)) - 1.
        
        print('numbtile')
        print(numbtile)

        # i indexes the y-axis
        for i in indxtile: 
            # j indexes the y-axis
            for j in indxtile: 
        
                print('Tile i=%d, j=%d' % (i, j))
                
                # plots
                ## file name string extension for image plots
                gdat.strgcntp = '%s_%02d%d%d_%d_%d' % (gdat.datatype, isec, icam, iccd, i, j)
               
                # determine the initial and final pixel indices of the FFI data to be copied
                indxsideyposdatainit = numbpixloffsypos + j * numbsidetile - gdat.numbsideedge
                indxsideyposdatafinl = numbpixloffsypos + (j + 1) * numbsidetile - gdat.numbsideedge
                indxsidexposdatainit = numbpixloffsxpos + i * numbsidetile - gdat.numbsideedge
                indxsidexposdatafinl = numbpixloffsxpos + (i + 1) * numbsidetile - gdat.numbsideedge
                if j == 0:
                    indxsideyposdatainit += gdat.numbneigback + gdat.offscorr
                    indxsideyposdatafinl += gdat.numbneigback + gdat.offscorr
                #if j == numbtile - 1:
                #    indxsideyposdatainit -= gdat.numbneigback + gdat.offscorr
                #    indxsideyposdatafinl -= gdat.numbneigback + gdat.offscorr
                if i == 0:
                    indxsidexposdatainit += gdat.numbneigback + gdat.offscorr
                    indxsidexposdatafinl += gdat.numbneigback + gdat.offscorr
                #if i == numbtile - 1:
                #    indxsidexposdatainit -= gdat.numbneigback + gdat.offscorr
                #    indxsidexposdatafinl -= gdat.numbneigback + gdat.offscorr
                
                #if gdat.strgmode == 'tcut':
                #    listhduncatl = listhdundata[o]
                #else:
                #    path = gdat.pathdataorig + listrtag[0]
                #    listhduncatl = fits.open(path)
                if gdat.datatype == 'obsd':
                    if gdat.strgmode == 'tcut':
                        cntptile = listhdundatatemp[1].data['FLUX'].astype(float) * timeexpo
                        cntptile = cntptile.swapaxes(0, 2)
                        cntptile = cntptile.swapaxes(0, 1)
                        cntptile = cntptile[:, :, ::30]

                    else:
                        pathsavedata = gdat.pathdata + '%s_data.npz' % gdat.strgcntp
                        if True or not os.path.exists(pathsavedata):
                            listobjtwcss = []
                            gdat.time = []
                            cntptile = []
                            for t in gdat.indxtime:
                                if t % 100 == 0:
                                    print('Loading the image into tilery, t = %d' % t)
                                
                                path = gdat.pathdataorig + listrtag[t]
                                listhdundatatemp = fits.open(path, memmap=False)
                                objtheadseco = listhdundatatemp[2].header
                                listobjtwcss.append(WCS(objtheadseco))
                                objtheadfrst = listhdundatatemp[0].header
                                timetemp = (objtheadfrst['TSTOP'] + objtheadfrst['TSTART']) / 2
                                gdat.time.append(timetemp)
                                
                                hdundata = listhdundatatemp[1].data.astype(float) * timeexpo
                                cntptile.append(hdundata[indxsideyposdatainit:indxsideyposdatafinl, indxsidexposdatainit:indxsidexposdatafinl])
                                listhdundatatemp.close()
                            
                            gdat.time = np.array(gdat.time)
                            cntptile = np.stack(cntptile, axis=-1)
                            listtemp = [listobjtwcss, gdat.time, cntptile]

                            objtfile = open(pathsavedata, 'wb')
                            
                            print('Writing to %s...' % pathsavedata)
                            pickle.dump(listtemp, objtfile)
                        else:
                            print('Reading from %s...' % pathsavedata)
                            objtfile = open(pathsavedata, "rb" )
                            listtemp = pickle.load(objtfile)
                            listobjtwcss, gdat.time, cntptile = listtemp
                       
                if gdat.datatype == 'mock':
                    print('Generating mock data...')
                    # Data generation
                    ## image
                    gdat.time = np.concatenate((np.linspace(1., 12.7, gdat.numbtime / 2), np.linspace(1., 12.7, gdat.numbtime / 2)))
                    arrytime = np.empty((2, gdat.numbtime))
                    arrytime[:, :] = np.linspace(-0.5, 0.5, gdat.numbtime)[None, :]
                    gdat.indxsour = np.arange(numbsour)
                    gdat.indxtime = np.arange(gdat.numbtime)
                    posiquat = 5e-2 * np.random.randn(2 * gdat.numbtime).reshape((2, gdat.numbtime)) + arrytime * 0.1
                    gdat.trueypos = numbsidetile * np.random.random(numbsour)[None, :] + posiquat[0, :, None]
                    gdat.truexpos = numbsidetile * np.random.random(numbsour)[None, :] + posiquat[1, :, None]
                    
                    cntptile = np.ones((numbsidetile, numbsidetile, gdat.numbtime)) * 6.
                    
                    # inject signal
                    indxsupn = np.arange(numbsupn)
                    truecntpsour = np.empty((gdat.numbtime, numbsour))
                    truemagt = np.empty((gdat.numbtime, numbsour))
                    gdat.indxsoursupn = np.random.choice(gdat.indxsour, size=numbsupn, replace=False)
                    for n in gdat.indxsour:
                        if n in gdat.indxsoursupn:
                            timenorm = -0.5 + (gdat.time / np.amax(gdat.time)) + 2. * (np.random.random(1) - 0.5)
                            objtrand = scipy.stats.skewnorm(10.).pdf(timenorm)
                            objtrand /= np.amax(objtrand)
                            truemagt[:, n] = 8. + 6. * (2. - objtrand)
                        else:
                            truemagt[:, n] = np.random.rand() * 5 + 15.
                        truecntpsour[:, n] = 10**((20.424 - truemagt[:, n]) / 2.5)
                    gdat.truemagtmean = np.mean(truemagt, 0)
                    gdat.truemagtstdv = np.std(truemagt, 0)

                    indxsideypossour = np.round(np.mean(gdat.trueypos, 0)).astype(int)
                    indxsidexpossour = np.round(np.mean(gdat.truexpos, 0)).astype(int)
                    
                    sigmpsfn = 1.

                    print('Generating synthetic count map for each source...')
                    for k in gdat.indxsour:
                        temp = 1. / np.sqrt(sigmpsfn**2 * (2. * np.pi)**2) * \
                            truecntpsour[None, None, :, k] * np.exp(-0.5 * ((indxsidexpostile[:, :, None] - gdat.truexpos[None, None, :, k])**2 + \
                                                                    (indxsideypostile[:, :, None] - gdat.trueypos[None, None, :, k])**2) / sigmpsfn**2)
                        cntptile[:, :, :] += temp
                    
                    indxsideypossour[np.where(indxsideypossour == numbsidetile)] = numbsidetile - 1
                    indxsidexpossour[np.where(indxsidexpossour == numbsidetile)] = numbsidetile - 1
                    indxsideypossour[np.where(indxsideypossour < 0)] = 0
                    indxsidexpossour[np.where(indxsidexpossour < 0)] = 0
                    
                    indxsourinsd = np.where((indxsideypossour > gdat.numbneigback + gdat.offscorr) & \
                                   (indxsidexpossour > gdat.numbneigback + gdat.offscorr) & (indxsideypossour < numbsidetile - gdat.numbsideedge) & \
                                   (indxsidexpossour < numbsidetile - gdat.numbsideedge))[0]
                    
                    indxsupninsd = np.where((indxsideypossour[gdat.indxsoursupn] > gdat.numbneigback + gdat.offscorr) & \
                                   (indxsidexpossour[gdat.indxsoursupn] > gdat.numbneigback + gdat.offscorr) & \
                                   (indxsideypossour[gdat.indxsoursupn] < numbsidetile - gdat.numbsideedge) & \
                                   (indxsidexpossour[gdat.indxsoursupn] < numbsidetile - gdat.numbsideedge))[0]
                    
                    indxdatasour = (indxsideypossour - gdat.numbsideedge) * numbsidesrch + indxsidexpossour - gdat.numbsideedge
                    indxdatasupn = (indxsideypossour[gdat.indxsoursupn] - gdat.numbsideedge) * numbsidesrch + \
                                                                        indxsidexpossour[gdat.indxsoursupn] - gdat.numbsideedge
                    
                    gdat.refrtmag = gdat.truemagtmean

                    indxdataback = np.setdiff1d(indxdata, indxdatasour)
                    listlabltrue[indxdatasour[indxsourinsd]] = 1
                    cntptile *= timeexpo
                    cntptile = np.random.poisson(cntptile).astype(float)
                
                    # temp
                    gdat.refrxpos = gdat.truexpos[0, :]
                    gdat.refrypos = gdat.trueypos[0, :]

                # rebin in time
                if gdat.boolrebn and gdat.numbtime > gdat.numbtimerebn:
                    print('Rebinning in time...')
                    numbtimeoldd = gdat.numbtime
                    gdat.numbtime = gdat.numbtimerebn
                    numbtimebins = numbtimeoldd / gdat.numbtime
                    cntptileneww = np.zeros((numbsidetile, numbsidetile, gdat.numbtime)) - 1.
                    timeneww = np.zeros(gdat.numbtime)
                    for t in range(gdat.numbtime):
                        if t == gdat.numbtime - 1:
                            cntptileneww[:, :, t] = np.mean(cntptile[:, :, (gdat.numbtime-1)*numbtimebins:], axis=2)
                            timeneww[t] = np.mean(gdat.time[(gdat.numbtime-1)*numbtimebins:])
                        else:
                            cntptileneww[:, :, t] = np.mean(cntptile[:, :, t*numbtimebins:(t+1)*numbtimebins], axis=2)
                            timeneww[t] = np.mean(gdat.time[t*numbtimebins:(t+1)*numbtimebins])
                    gdat.indxtimegood = np.isfinite(timeneww)
                    cntptile = cntptileneww[:, :, gdat.indxtimegood]
                    gdat.time = timeneww[gdat.indxtimegood]
                    gdat.numbtime = gdat.indxtimegood.size
                    gdat.indxtime = np.arange(gdat.numbtime)
                
                print('cntptile')
                summgene(cntptile)
                
                # check data
                if cntptile.shape[0] != numbsidetile or cntptile.shape[1] != numbsidetile:
                    print('cntptile.shape')
                    print(cntptile.shape)
                    print('numbsidetile')
                    print(numbsidetile)
                    raise Exception('')

                # check cntptile
                if not np.isfinite(cntptile).all():
                    raise Exception('')

                # string of sector, camera, CCD and tile
                gdat.strgtitl = 'Sector %d, Camera %d, CCD %d, Tile %d%d' % (isec, icam, iccd, i, j)
                
                # plot the temporal median of the tile data
                cntptiletmed = np.median(cntptile, axis=2)
                plot_imag(gdat, cntptiletmed, 'cntptiletmed', cmap='Greys_r', strgtitlinit='Temporal median\n')
                
                # initialize the background
                cntpback = np.empty_like(cntptile)
                print('cntpback')
                summgene(cntpback)

                if i == 0 and j == 0 and gdat.boolquatplot:
                    print('Reading the quaternion...')
                    pathbasequat = os.environ['TCAT_DATA_PATH'] + '/data/quat/'
                    path = pathbasequat + fnmatch.filter(os.listdir(pathbasequat), 'tess*_sector%02d-quat.fits' % isec)[0]
                    listhdunquat = fits.open(path)
                    # get data
                    timequat = listhdunquat['CAMERA%d' % listicam[o]].data.field('TIME')[10000:20000]
                    quat = np.empty((timequat.size, 2))
                    quat[:, 0] = listhdunquat['CAMERA%d' % listicam[o]].data.field('C%d_Q1_SC' % listicam[o])[10000:20000]
                    quat[:, 1] = listhdunquat['CAMERA%d' % listicam[o]].data.field('C%d_Q2_SC' % listicam[o])[10000:20000]
                    
                    # plot the quaternion
                    numb = len(listhdunquat['CAMERA%d' % listicam[o]].data[0])
                    for k in range(numb):
                        figr, axis = plt.subplots()
                        axis.hist(listhdunquat['CAMERA%d' % listicam[o]].data.field(k), bins=100)
                        axis.set_yscale('log')
                        axis.set_xlabel(listhdunquat['CAMERA%d' % listicam[o]].columns.names[k])
                        path = gdat.pathimag + 'quat_%04d.%s' % (k, gdat.strgplotextn)
                        plt.savefig(path)
                        plt.close()
                    listhdunquat.close()
                    
                    # plot quats
                    figr, axis = plt.subplots(2, 1, sharex='all')
                    axis[0].plot(timequat, quat[:, 0] * 3600., alpha=0.4, color='b', ls='', marker='o', markersize=3, label='q_x')
                    axis[1].plot(timequat, quat[:, 1] * 3600., alpha=0.4, color='g', ls='', marker='o', markersize=3, label='q_y')
                    for a in range(2):
                        if a == 0:
                            delt = 1. / 60. / 24.
                        if a == 1:
                            delt = 1. / 24.
                        binstimequat = np.arange(np.amin(timequat), np.amax(timequat), delt)
                        meantimequat = (binstimequat[1:] + binstimequat[:-1]) / 2.
                        numbtimequatrebn = meantimequat.size
                        
                        quatrebn = np.empty((numbtimequatrebn, 2))
                        for k in range(numbtimequatrebn):
                            indx = np.where((timequat > binstimequat[k]) & (timequat < binstimequat[k+1]))[0]
                            for b in range(2):
                                quatrebn[k, b] = np.mean(quat[indx, b])
                        axis[0].plot(meantimequat, quatrebn[:, 0], alpha=0.4, color='r', ls='', marker='o', markersize=3)
                        axis[1].plot(meantimequat, quatrebn[:, 1], alpha=0.4, color='y', ls='', marker='o', markersize=3)
                    for b in range(2):
                        pass
                        #axis[b].legend()
                    axis[1].set_xlabel('Time [BJD]')
                    axis[1].set_ylabel('Quaternion [arcsec]')
                    path = gdat.pathimag + 'quat_%s.png' % (gdat.strgcntp)
                    plt.savefig(path)
                    plt.close()
                
                if gdat.boolquat:
                    print('Shifting the images based on the mock quaternion...')
                    print('quat')
                    summgene(quat)
                    # temp
                    f = interpolate.interp2d(quat[0, :], quat[1, :], cntptile[:, :, 0], kind='cubic')
                    for t in gdat.indxtime:
                        cntptile[:, :, t] = f(x + quat[0, t], y + quat[1, t])
                
                # plot the light curve of the spatial median of the background
                cntptilesmed = np.median(cntptile, (0, 1))
                figr, axis = plt.subplots()
                axis.plot(gdat.time, cntptilesmed, color='black', ls='', marker='o', markersize=3)
                path = gdat.pathimag + 'cntptilesmed_%s.%s' % (gdat.strgcntp, gdat.strgplotextn)
                plt.savefig(path)
                plt.close()
                
                gdat.ramptime = np.linspace(0., 1., gdat.numbtime)
                
                print('Determining the spatially smooth background...')
                #listsizefilt = [11, 15, 19]
                listsizefilt = [15]
                for sizefilt in listsizefilt:
                    if (i != 0 or j != 0) and sizefilt != listsizefilt[len(listsizefilt)/2]:
                        continue
                    cntpbacktemp = np.empty_like(cntpback)
                    for t in gdat.indxtime:
                        cntpbacktemp[:, :, t] = scipy.ndimage.median_filter(cntptile[:, :, t], size=sizefilt)
                    if sizefilt == listsizefilt[len(listsizefilt)/2]:
                        cntpback = cntpbacktemp
                    plot_anim(gdat, cntpbacktemp, 'cntpback%04d' % sizefilt, strgtitlextn=', Kernel = %d' % sizefilt)
                    
                print('Taking out the spatially smooth background...')
                cntptilepnts = cntptile - cntpback
                cntptilepntstmed = np.median(cntptilepnts, 2)
                
                print('Masking out bright pixels...')
                #if gdat.boolcatl:
                #    gdat.indxsourbrgt = np.where(gdat.refrtmag < 12)[0]
                #cntpdatatilemskd = np.copy(cntptile)
                #if gdat.boolcatl:
                #    for k in gdat.indxsourbrgt:
                #        dist = (xpospixlmesh - gdat.refrxpos[k])**2 + (ypospixlmesh - gdat.refrypos[k])**2
                #        indxxposmask, indxyposmask = np.where(dist < gdat.distsqrd)
                #        cntpdatatilemskd[indxyposmask, indxxposmask, :] = np.amin(cntptilepnts)
                listperc = [50, 90, 99]
                for perc in listperc:
                    cntpdatatilemskd = np.copy(cntptilepnts)
                    cntpdatatilemskd[np.where(cntptilepnts > np.percentile(cntptilepnts, perc))] = np.amin(cntptilepnts)
                    plot_anim(gdat, cntpdatatilemskd, 'cntpdatatilemskd%04d' % perc, strgtitlextn=', Percentile = %d' % perc)
                
                #cntpcutt = 100 * timeexpo
                #gdat.indxtimeorbt = np.argmax(np.diff(gdat.time))
                #gdat.indxtimegood = np.where((gdat.time > np.amin(gdat.time) + 2.) & \
                #                        (gdat.time < np.amax(gdat.time) - 0.) & (cntptilesmed < cntpcutt) & \
                #                        ((gdat.time < gdat.time[gdat.indxtimeorbt] - 0.) | (gdat.time > gdat.time[gdat.indxtimeorbt+1] + 2.)))[0]
                #print('Number of pixels that pass the mask: %d' % gdat.indxtimegood.size)
                #if gdat.datatype == 'obsd':
                #    listobjtwcss = np.array(listobjtwcss)[gdat.indxtimegood]
                #gdat.time = gdat.time[gdat.indxtimegood]
                #cntptile = cntptile[:, :, gdat.indxtimegood]
                #gdat.numbtime = gdat.time.size
                #gdat.indxtime = np.arange(gdat.numbtime)
                #cntptilesmed = cntptilesmed[gdat.indxtimegood]
                
                listtimefilt = [10, 100, 300]
                for timefilt in listtimefilt:
                    # temporal median filter
                    numbtimefilt = min(9, gdat.numbtime)
                    if numbtimefilt % 2 == 0:
                        numbtimefilt -= 1
                    print('Performing the temporal median filter...')
                    cntptiletflt = scipy.signal.medfilt(cntptilepnts, (1, 1, numbtimefilt))
                    
                    cntptilefinl = cntptilepnts - cntptiletflt
                    plot_anim(gdat, cntptilefinl, 'cntptilefinl%04d' % timefilt, strgtitlinit='Temporally filtered count map, ', \
                                                                                                    strgtitlextn=', Time Kernel Size = %d' % timefilt)
                    
                    # calculate derived maps
                    ## RMS image
                    stdvcntppnts = np.std(cntptilefinl, 2)# / np.median(cntptilefinl, 2)
                    plot_imag(gdat, stdvcntppnts, 'stdvcntppnts%04d' % timefilt, cmap='Reds', strgtitlinit='RMS, %s' % gdat.strgtitl)
            
                    # sort the pixels with respect to RMS
                    indxpixlsort = np.argsort(stdvcntppnts.flatten())
                    
                    indxpixlsort = np.argsort(np.std(cntpdatatilemskd, 2).flatten())[::-1]
                    
                    # take the top N pixels
                    indxpixlsort = indxpixlsort[:gdat.numbpixlposi]

                    gdat.indxxposposisort = gdat.indxsidexpos.flatten()[indxpixlsort]
                    gdat.indxyposposisort = gdat.indxsideypos.flatten()[indxpixlsort]
                    
                    print('')
                    print('gdat.indxxposposisort')
                    print(gdat.indxxposposisort)

                    # find the sky positions of the pixels
                    print('listobjtwcss')
                    print(listobjtwcss)
                    posiskyy = listobjtwcss[0].all_pix2world(gdat.indxyposposisort, gdat.indxxposposisort, 0)
                    rascposipixl = np.empty(gdat.numbpixlposi)
                    declposipixl = np.empty(gdat.numbpixlposi)
                    gdat.g2idposipnts = [[] for n in gdat.indxpixlposi]
                    ticiposipnts = [[] for n in gdat.indxpixlposi]
                    rascposipnts = [[] for n in gdat.indxpixlposi]
                    declposipnts = [[] for n in gdat.indxpixlposi]
                    for n in gdat.indxpixlposi:
                        strgsrch = '%g %g' % (posiskyy[0][n], posiskyy[1][n])
                        rascposipixl[n] = posiskyy[0][n]
                        declposipixl[n] = posiskyy[1][n]
                        catalogData = Catalogs.query_region(strgsrch, radius='0.5m', catalog = "Gaia")
                        if len(catalogData) > 0:
                            #ticiposipnts[n] = int(catalogData[0]['ID'])
                            gdat.g2idposipnts[n] = int(catalogData[0]['source_id'])
                            rascposipnts[n] = catalogData[:]['ra']
                            declposipnts[n] = catalogData[:]['dec']
                    
                    plot_imag(gdat, stdvcntppnts, 'stdvcntppnts%04d' % timefilt, cmap='Reds', strgtitlinit='RMS, %s' % gdat.strgtitl, boolcatl=True)
                    
                    #posiskyy = np.empty((rasctarg.size, 2))
                    #posiskyy[:, 0] = rasctarg
                    #posiskyy[:, 1] = decltarg
                    #posipixl = listobjtwcss[0].all_world2pix(posiskyy, 0)
                    #gdat.refrxpos = posipixl[:, 0] 
                    #gdat.refrypos = posipixl[:, 1] 
                    #gdat.refrtmag = catalogData[:]['Tmag']
                
                    arrytemp = np.zeros_like(cntptile[:, :, 0])
                    arrytemp[gdat.indxyposposisort, gdat.indxxposposisort] = 1.
                    plot_imag(gdat, arrytemp, 'arrytemp%04d' % timefilt, cmap='Reds', strgtitlinit='RMS, %s' % gdat.strgtitl)

                    ## light curves
                    figr, axis = plt.subplots(gdat.numbpixlposi, 1, figsize=(12, 20))
                    for p in gdat.indxpixlposi:
                        axis[p].plot(gdat.time, cntptilepnts[gdat.indxyposposisort[p], gdat.indxxposposisort[p], :], \
                                                                                        color='black', ls='', marker='o', markersize=3)
                        if p != 9:
                            axis[9].set_xticks([])
                        axis[p].set_xlabel('ADU')
                    axis[9].set_xlabel('Time')
                    path = gdat.pathimag + 'lcur_%s.%s' % (gdat.strgcntp, gdat.strgplotextn)
                    plt.savefig(path)
                    plt.close()
    
                if False:
                    if i == 0 and j == 0:
                        lcurarry = np.empty((numbsidesrch, numbsidesrch, gdat.numbtime, numbsidecorr, numbsidecorr)) 
                        medilcur = np.empty((numbsidesrch, numbsidesrch)) 
               
                    lcuravgd = np.empty(gdat.numbtime)
                    cntr = 0
                    prevfrac = -1
                    k = 0
                    
                    # machine learning
                    ## get neighboring pixels as additional features
                    for a in np.arange(numbsidecorr):
                        for b in np.arange(numbsidecorr):
                            if numbsidecorr == 3:
                                offs = -1
                            else:
                                offs = 0
                            indx = gdat.numbneigback + gdat.offscorr + a + offs
                            lcurarry[:, :, :, a, b] = cntptile[indx:indx+numbsidesrch, indx:indx+numbsidesrch, :]
                    lcurflat = lcurarry.reshape((numbsidesrch**2, gdat.numbtime, numbsidecorr, numbsidecorr))
                    lcurflat = lcurflat.reshape((numbsidesrch**2, gdat.numbtime * numbsidecorr**2))

                    strgtype = 'tsne'
                            
                    indxsideypostemp = k * numbstrd
                    indxsidexpostemp = l * numbstrd
                    if not np.isfinite(cntptile).all():
                        raise Exception('')

                    if strgtype == 'tsne' or strgtype == 'tmpt':
                        pass
                    else:
                        indxsideyposaperinit = indxsideypostemp - numbneigaper
                        indxsidexposaperinit = indxsidexpostemp - numbneigaper
                        indxsideyposaperfinl = indxsideypostemp + numbneigaper + 1
                        indxsidexposaperfinl = indxsidexpostemp + numbneigaper + 1
                        indxsideyposbackinit = indxsideypostemp - gdat.numbneigback
                        indxsidexposbackinit = indxsidexpostemp - gdat.numbneigback
                        indxsideyposbackfinl = indxsideypostemp + gdat.numbneigback + 1
                        indxsidexposbackfinl = indxsidexpostemp + gdat.numbneigback + 1
                        
                        imagbackmedi = np.median(cntptile[indxsideyposbackinit:indxsideyposbackfinl, \
                                                                indxsidexposbackinit:indxsidexposbackfinl, :], axis=(0, 1))
                        for t in gdat.indxtime:
                            lcur[t] = np.sum(cntptile[indxsideyposaperinit:indxsideyposaperfinl, indxsidexposaperinit:indxsidexposaperfinl, t]) - \
                                                                                                        imagbackmedi[t] * numbpixlaper
                        if not np.isfinite(lcur).all():
                            raise Exception('')
                        
                        # normalize
                        meanlcur = np.mean(lcur)
                        lcur /= meanlcur
                        #lcurmedi = scipy.signal.medfilt(lcur, 11)
                        dictpara['mean'][k, l] = meanlcur
                        
                        lcurdiff = lcur - lcuravgd
                        gdat.indxtimediff = np.argsort(lcurdiff)[::-1]
                        for t in gdat.indxtimediff:
                            if t < 0.2 * gdat.numbtime or (t >= 0.5 * gdat.numbtime and t <= 0.7 * gdat.numbtime):
                                continue
                            if lcurdiff[t] > thrsdiff and lcurdiff[t-1] > thrsdiff:
                                break
                        gdat.indxtimediffaccp = t
                        dictpara['diff'][k, l] = lcurdiff[gdat.indxtimediffaccp]

                        # acceptance condition
                        boolgood = False
                        if dictpara['diff'][k, l] > thrsdiff:
                        #if dictpara['maxm'][k, l] > thrsmaxm and (lcurtest[gdat.indxtimemaxm+1] > thrsmaxm or lcurtest[gdat.indxtimemaxm-1] > thrsmaxm):
                            boolgood = True
                        
                        if abs(k - 50) < 4 and abs(l - 50) < 4:
                            boolgood = True
                            
                        if boolgood:# or (abs(k - 53) < 5 and abs(l - 53) < 5):
                            
                            indxsideyposaccp.append(k * numbstrd)
                            indxsidexposaccp.append(l * numbstrd)
                    
                            # plot
                            figr, axis = plt.subplots(figsize=(12, 6))
                            axis.plot(time, lcur, ls='', marker='o', markersize=3, label='Raw')
                            #axis.plot(time, lcurmedi, ls='', marker='o', markersize=3, label='Median')
                            #axis.plot(time, lcurtest, ls='', marker='o', markersize=3, label='Cleaned')
                            axis.plot(time, lcurdiff + 1., ls='', marker='o', markersize=3, label='Diff')
                            
                            axis.set_xlabel('Time [days]')
                            axis.set_ylabel('Relative Flux')
                            axis.legend()
                            
                            axis.axhline(thrsdiff + 1., ls='--', alpha=0.3, color='gray')
                            axis.axvline(time[int(0.2*gdat.numbtime)], ls='--', alpha=0.3, color='red')
                            axis.axvline(time[int(0.5*gdat.numbtime)], ls='--', alpha=0.3, color='red')
                            axis.axvline(time[int(0.7*gdat.numbtime)], ls='--', alpha=0.3, color='red')
                            
                            posisili = np.empty((1, 2))
                            posisili[0, 0] = indxsideyposdatainit + k
                            posisili[0, 1] = indxsidexposdatainit + l
                            
                            titl = 'Diff: %g' % (dictpara['diff'][k, l])
                            if gdat.datatype == 'obsd':
                                if pathfile is None:
                                    posiskyy = listobjtwcss[t].all_pix2world(posisili, 0)
                                else:
                                    posiskyy = listobjtwcss.all_pix2world(posisili, 0)
                                rasc = posiskyy[:, 0]
                                decl = posiskyy[:, 1]
                                strgsrch = '%g %g' % (rasc, decl)
                                titl += 'k = %d, l = %d' % (k, l)
                                #catalogData = Catalogs.query_region(strgsrch, radius='0.1m', catalog = "TIC")
                                #if len(catalogData) > 0:
                                #    tici = int(catalogData[0]['ID'])
                                #    titl += ', TIC %d' % tici
                            
                                #axis.axvspan(1345, 1350, alpha=0.5, color='red')
                            
                            axis.set_title(titl)
                            plt.tight_layout()
                            path = gdat.pathimag + 'lcur_%s_%04d_%04d.%s' % (gdat.strgcntp, k, l, gdat.strgplotextn)
                            print('Writing to %s...' % path)
                            plt.savefig(path)
                            plt.close()
                    


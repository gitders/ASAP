import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bruges as b
from scipy.interpolate import interp1d

color_top = 'darkcyan'
color_top_name = '.4'
alpha_top = 0.5
topbox = dict(boxstyle='round', ec='none', fc='w', alpha=0.7)
format_tops={'fontsize':8, 'color':color_top_name, 'ha':'right', 'bbox':topbox}
format_title={'fontsize':12, 'y':0.03, 'x':0.98, 'ha':'right', 'weight':'bold'}
format_title_topright={'fontsize':14, 'y':0.98, 'x':0.98, 'ha':'right', 'weight':'bold'}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
def get_well_files(wells_dir, name):
   well_files = []
   for dirpath, _, filenames in os.walk(wells_dir):
       well_files += [os.path.join(dirpath, f) for f in filenames if f.startswith(name)]
   return well_files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_nears_fars(df):
   sample_size = 64
   number_of_splits = abs(df['NEAR'].size/64)
   nears = np.array_split(df['NEAR'].values, number_of_splits)
   fars = np.array_split(df['FAR'].values, number_of_splits)
   nears = np.asarray(nears).transpose()
   fars = np.asarray(fars).transpose()
   return nears, fars

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_twt(tdr,z):
    tt=tdr[:,1]
    zz=tdr[:,0] 
    d2t = interp1d(zz, tt, kind='linear', bounds_error=False, fill_value='extrapolate')
    return d2t(z)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_synt(WW,ang,wavelet,method='shuey'):
    '''
    WW: Pandas dataframe with VP, VS, RHO (optionally EPSILON and DELTA)
    ang: angle range, define with ang=np.arange(0,50,1)
    wavelet
    method: 'shuey' (Shuey 2-terms), 'shuey3' (Shuey 3-terms),
            'aki' (Aki-Richards)
    '''
    WW.columns=WW.columns.str.upper()
    uvp, lvp   = WW.VP.values[:-1], WW.VP.values[1:]
    uvs, lvs   = WW.VS.values[:-1], WW.VS.values[1:]
    urho, lrho = WW.RHO.values[:-1], WW.RHO.values[1:]
    z=WW.index.values  # z is two-way-time
    synt  = np.zeros((z.size,ang.size))
    #--> calculate reflectivities with AVO equation,
    #--> convolve with input wavelet and fill in traces of synthetic seismogram
    for i,alpha in enumerate(ang):
        if method is 'shuey':
            RC = shuey(uvp,uvs,urho,lvp,lvs,lrho,alpha)
        elif method is 'shuey3':
            RC = shuey(uvp,uvs,urho,lvp,lvs,lrho,alpha,approx=False)
        else:
            RC = akirichards(uvp,uvs,urho,lvp,lvs,lrho,alpha)
        RC = np.append(np.nan, RC)
        RC = np.nan_to_num(RC)
        synt[:,i] = np.convolve(RC, wavelet, mode='same')
    return RC, synt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_synt(WW,synt,ztop,zbot,gain=10):
    '''
    WW: Pandas dataframe with VP, VS, RHO (in time)
    synt: synthetic seismogram computed with make_synt (in time)
    ztop,zbot: display window
    gain: multiplier to be applied to wiggles (default=5)
    method: 'shuey' (Shuey 2-terms), 'shuey3' (Shuey 3-terms),
            'aki' (Aki-Richards)
    '''
    WW.columns=WW.columns.str.upper()
    it1=np.abs(WW.index-ztop).argmin()
    it2=np.abs(WW.index-zbot).argmin()
    ss = synt[it1:it2,:]
    clip=np.abs(synt.max())
    f,ax=plt.subplots(nrows=1,ncols=5)
    opz1={'color':'k','linewidth':.5}
    opz2={'linewidth':0, 'alpha':0.6}
    ax[0].plot(WW.VP*WW.RHO,WW.index,'-k')
    ax[1].plot(WW.VP/WW.VS,WW.index,'-k')
    ax[2].plot(synt[:,0],WW.index,'-k')
    ax[3].plot(synt[:,1],WW.index,'-k')
    im=ax[4].imshow(ss,interpolation='none', cmap='Greys',aspect='auto')
    cbar=plt.colorbar(im, ax=ax[4])
    ax[0].set_xlabel('AI [m/s*g/cc]')
    ax[0].set_ylabel('TWT [s]')
    ax[1].set_xlabel('Vp/Vs')
    ax[2].set_title('Near')
    ax[3].set_title('Far')
    ax[4].set_title('Near|Far')
    for aa in ax[:4]:
        aa.set_ylim(zbot,ztop)
        aa.grid()
    for aa in ax[1:]:
        aa.set_yticklabels([])
    for aa in ax[2:4]:
        aa.set_xlim(-clip,+clip)
        aa.set_xticklabels([])
    for aa in ax[:2]:
        aa.xaxis.tick_top()
        plt.setp(aa.xaxis.get_majorticklabels(), rotation=90)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def td(WW,sr=0.1524,KB=0,WD=0,repl_vel=1600):
    '''
    td (C) aadm 2016
    Calculates time-depth relation by sonic integration.

    INPUT
    WW: Pandas dataframe
    sr: depth sampling rate (m, default 0.1524 m = half-foot)
    KB: kelly bushing elevation (m, default 0)
    WD: water depth (m, default 0)
    repl_vel: replacement velocity for overburden i.e. interval between seafloor and beginning of the logs
              (m/s, default 1600)

    OUTPUT
    numpy array with 2 columns: 0=depth, 1=twt (secs)
    '''
    WW.columns=WW.columns.str.upper()
    if 'TVD' in WW.columns:
        depth = WW.TVD.values
    else:
        depth = WW.index.values
    WW.VP.interpolate(inplace=True)
    sonic=1/WW.VP.values # VP in m/s
    start = depth.min()
    water_vel = 1480
    wb_twt = 2.0*WD/water_vel
    sonic_start=depth[np.isfinite(sonic)].min()
    sonic_start_twt=2*(sonic_start-KB-WD)/repl_vel + wb_twt
    scaled_sonic = sr*sonic[depth>=sonic_start]
    twt = 2*np.cumsum(scaled_sonic) + sonic_start_twt
    print('[TD] water bottom two-way-time: {:.3f} [s]'.format(wb_twt))
    print('[TD] sonic log start: {:.3f} [m] = {:.3f} [s]'.format(sonic_start, sonic_start_twt))
    print('[TD] computed twt scale range: {:.3f}-{:.3f} [s]'.format(twt.min(),twt.max()))
    return np.column_stack((depth[depth>=sonic_start],twt))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def welltime(WW,tdr,dt=0.001,ztop=None,zbot=None,name=None,tops=None,qcplot=True):
    '''
    welltime (C) aadm 2016
    Converts logs sampled in depth to time using a reference time-depth function.
    Load existing t-d with tdr=np.loadtxt('TD.dat', delimiter=',')
    or use squit.well.td to create one.

    INPUT
    WW: Pandas dataframe
    tdr: time-depth table: numpy array shape Nx2, column 0 = MD [m], column 1 = TWT [s]
    dt: sample rate in seconds
    ztop,zbot: display window (defaults to min,max depth)
    name: well name (or anything else) to print
    tops: dictionary containing stratigraphic tops, e.g.: tops={'Trias': 1200,'Permian': 2310}
          or Pandas Series, e.g: tops=pd.Series({'Trias': 1200,'Permian': 2310})

    OUTPUT
    Pandas dataframe with logs sampled in time
    ztop, zbot converted to TWT
    '''
    WW.columns=WW.columns.str.upper()
    flagfrm=True if 'VP_FRMB' in WW.columns else False
    z = WW.index
    if ztop is None: ztop = z.min()
    if zbot is None: zbot = z.max()
    flagtops=False if tops is None else True
    if flagtops:
        if not isinstance(tops, pd.Series):
            tops=pd.Series(tops)
        tops=tops.dropna().sort_values()

    #--> load depth-time relationship (depth is MD)
    tt=tdr[:,1]
    zz=tdr[:,0]
    #-->  twt reference log sampled like depth reference log
    # twt_log = np.interp(z, zz, tt, left=np.nan, right=np.nan)
    ff=(z>=zz.min()) & (z<=zz.max())
    twt_log = np.interp(z[ff], zz, tt, left=np.nan, right=np.nan)
    #-->  interpolant to convert depths to times on the fly (e.g., tops)
    d2t = interp1d(zz, tt, kind='linear', bounds_error=False, fill_value='extrapolate')
    if qcplot:
        print('[WELLTIME] plot window top, bottom [m]:{:.0f}-{:.0f}, [s]:{:.4f}-{:.4f}'.format(ztop,zbot,float(d2t(ztop)),float(d2t(zbot))))
    #-->  regularly-sampled twt scale and its depth (MD) equivalent on the basis of depth-time rel.
    twt = np.arange(0, tt.max(), dt)
    zt = np.interp(x=twt, xp=tt, fp=zz, left=np.nan, right=np.nan)

    #-->  resample logs to twt
    WWt=pd.DataFrame(data=zt, columns=['DEPTH'], index=twt)
    WWt.index.rename('TWT',inplace=True)
    loglist=WW.columns
    for i in loglist:
        tmp = np.interp(x=twt, xp=twt_log, fp=WW[i][ff].values, left=np.NaN, right=np.NaN)
        WWt=pd.concat([WWt,pd.Series(tmp, index=twt, name=i)],axis=1)
        WWt.interpolate(inplace=True)
        WWt.fillna(method = 'bfill',inplace=True)

    #--> QC plot with IP in depth and time
    if qcplot:
        tmp_IP = WW['VP']*WW['RHO']
        tmp_IPt = WWt['VP']*WWt['RHO']
        plotmax = tmp_IP[(z>=ztop) & (z<=zbot)].max()
        plotmin = tmp_IP[(z>=ztop) & (z<=zbot)].min()
        plotmax += plotmax*.1
        plotmin -= plotmin*.1

        f, ax = plt.subplots(nrows=1,ncols=2,figsize=(5,5), facecolor='w')
        ax[0].plot(tmp_IP, z, '-k')
        ax[1].plot(tmp_IPt, WWt.index, '-k')
        ax[0].set_xlabel('AI [m/s*g/cc]'), ax[0].set_ylabel('MD [m]')
        ax[1].set_xlabel('AI [m/s*g/cc]'), ax[1].set_ylabel('TWT [s]')
        ax[1].yaxis.set_label_position('right')
        ax[1].yaxis.set_ticks_position('right')
        ax[0].set_ylim(ztop,zbot)
        ax[1].set_ylim(d2t(ztop),d2t(zbot))
        for aa in ax.flatten():
            aa.invert_yaxis()
            aa.grid()
            aa.xaxis.tick_top()
            plt.setp(aa.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
            # aa.set_xlim(plotmin,plotmax)
        if flagtops: # plot top markers on all columns
            for topn,topz in tops.iteritems():
                if (topz>=ztop) & (topz<=zbot):
                    ax[0].axhline(y=topz,color=color_top,alpha=alpha_top)
                    ax[0].text(x=plotmax,y=topz,s=topn,**format_tops)
                    ax[1].axhline(y=d2t(topz),color=color_top,alpha=alpha_top)
                    ax[1].text(x=plotmax,y=d2t(topz),s=topn,**format_tops)
        if name is not None:
            plt.suptitle(name, **format_title)
        # plt.subplots_adjust(right=0.7,bottom=0.15)
        plt.tight_layout()
    return WWt,float(d2t(ztop)),float(d2t(zbot))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta, approx=True, terms=False):
    '''
    shuey (C) aadm 2016
    Calculates P-wave reflectivity with Shuey's equation

    reference:
    Avseth et al. (2005), Quantitative Seismic Interpretation, Cambridge University Press (p.182)

    INPUT
    vp1, vs1, rho1: P-, S-wave velocity (m/s) and density (g/cm3) of upper medium
    vp2, vs2, rho2: P-, S-wave velocity (m/s) and density (g/cm3) of lower medium
    theta: angle of incidence (degree)
    approx: returns approximate (2-term) form (default: True)
    terms: returns reflectivity, intercept and gradient (default: False)

    OUTPUT
    reflectivity (and optionally intercept, gradient; see terms option) at angle theta
    '''
    a = np.radians(theta)
    dvp = vp2-vp1
    dvs = vs2-vs1
    drho = rho2-rho1
    vp  = np.mean([vp1,vp2])
    vs  = np.mean([vs1,vs2])
    rho = np.mean([rho1,rho2])
    R0 = 0.5*(dvp/vp + drho/rho)
    G  = 0.5*(dvp/vp) - 2*(vs**2/vp**2)*(drho/rho+2*(dvs/vs))
    F =  0.5*(dvp/vp)
    if approx:
        R = R0 + G*np.sin(a)**2
    else:
        R = R0 + G*np.sin(a)**2 + F*(np.tan(a)**2-np.sin(a)**2)
    if terms:
        return R,R0,G
    else:
        return R

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta):
    '''
    Aki-Richards (C) aadm 2017
    Calculates P-wave reflectivity with Aki-Richards approximate equation
    only valid for small layer contrasts.

    reference:
    Mavko et al. (2009), The Rock Physics Handbook, Cambridge University Press (p.182)

    INPUT
    vp1, vs1, rho1: P-, S-wave velocity (m/s) and density (g/cm3) of upper medium
    vp2, vs2, rho2: P-, S-wave velocity (m/s) and density (g/cm3) of lower medium
    theta: angle of incidence (degree)

    OUTPUT
    reflectivity at angle theta
    '''
    a = np.radians(theta)
    p = np.sin(a)/vp1
    dvp = vp2-vp1
    dvs = vs2-vs1
    drho = rho2-rho1
    vp  = np.mean([vp1,vp2])
    vs  = np.mean([vs1,vs2])
    rho = np.mean([rho1,rho2])
    A = 0.5*(1-4*p**2*vs**2)*drho/rho
    B = 1/(2*np.cos(a)**2) * dvp/vp
    C = 4*p**2*vs**2*dvs/vs
    R = A + B - C
    return R

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def quicklook(WW,ztop=None,zbot=None,name=None,tops=None,linethickness=1):
    '''
    quicklook (C) aadm 2015-2018
    Summary well plot with raw and processed logs.

    INPUT
    WW: Pandas dataframe with VP, [VS], RHO, IP, [VPVS or PR], [SWE], PHIE, VSH
    ztop,zbot: depth range to plot (defaults to min,max depth)
    name: well name (or anything else) to print
    tops: dictionary containing stratigraphic tops, e.g.: tops={'Trias': 1200,'Permian': 2310}
          or Pandas Series, e.g: tops=pd.Series({'Trias': 1200,'Permian': 2310})
    '''
    WW.columns=WW.columns.str.upper()
    z=WW.index
    if ztop is None: ztop = z.min()
    if zbot is None: zbot = z.max()
    if 'PHIE' not in WW.columns:
        nocpi=True
    else:
        nocpi=False
        # if 'VOL_UWAT' not in WW.columns:
        #     WW['VOL_UWAT'] = WW.PHIE.values*WW.SWE.values
        if 'VSH' not in WW.columns:
            WW['VSH']=WW.VCL
    flagvs=True if 'VS' in WW.columns else False
    flagtops=False if tops is None else True
    if flagtops:
        if not isinstance(tops, pd.Series):
            tops=pd.Series(tops)
        tops=tops.dropna().sort_values()

    if flagvs:
        velmin=WW.VS[(z>=ztop) & (z<=zbot)].min()
        rmin=WW.VPVS[(z>=ztop) & (z<=zbot)].min()
        rmax=WW.VPVS[(z>=ztop) & (z<=zbot)].max()
    else:
        velmin=WW.VP[(z>=ztop) & (z<=zbot)].min()
        rmin=0.0
        rmax=5.0
    velmax=WW.VP[(z>=ztop) & (z<=zbot)].max()
    dmin=WW.RHO[(z>=ztop) & (z<=zbot)].min()
    dmax=WW.RHO[(z>=ztop) & (z<=zbot)].max()
    ipmin=WW.IP[(z>=ztop) & (z<=zbot)].min()
    ipmax=WW.IP[(z>=ztop) & (z<=zbot)].max()
    grmin=WW.GR[(z>=ztop) & (z<=zbot)].min()
    grmax=WW.GR[(z>=ztop) & (z<=zbot)].max()

    f, ax = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(8,5))
    if nocpi:
        ax[0].plot(WW.GR, z, color='.2', alpha=1.0, lw=1, label='GR')
        ax[0].fill_betweenx(z,grmin,WW.GR, where=None, facecolor='.5', linewidth=0, alpha=0.5)
        ax[0].fill_betweenx(z,WW.GR,grmax, where=None, facecolor='yellow', linewidth=0, hatch='...')
        ax[0].set_xlim(grmin,grmax)
        ax[0].set_ylim(zbot,ztop)
        ax[0].set_xlabel('GR')
    else:
        axphi = ax[0].twiny()
        axphi.plot(WW.PHIE, z, color='black', alpha=0.8, lw=2)
        # axphi.fill_betweenx(z,WW.PHIE,WW.VOL_UWAT, where=WW.VOL_UWAT>0, facecolor='red', linewidth=0, alpha=0.5)
        # axphi.fill_betweenx(z,WW.VOL_UWAT,0, where=WW.VOL_UWAT>0, facecolor='cyan', linewidth=0, alpha=0.5)
        axphi.set_xlim(1.1,-0.1)
        axphi.set_ylim(zbot,ztop)
        axphi.xaxis.tick_bottom()
        axphi.xaxis.set_label_position('bottom')
        axphi.set_xlabel('phi', color='k', weight='bold')
        plt.setp(axphi.xaxis.get_majorticklabels(), fontsize=8, color='k')
        # ax[0].plot(WW.VSH, z, color='.5', alpha=1.0, lw=1, label='Vsh')
        # ax[0].fill_betweenx(z,WW.VSH,0, where=WW.VSH>0, facecolor='.5', linewidth=0, alpha=0.5)
        # ax[0].fill_betweenx(z,1-WW.PHIE,WW.VSH,where=WW.VSH>0, facecolor='yellow', linewidth=0, hatch='...')
        ax[0].set_xlim(-0.1,1.1)
        ax[0].set_ylim(zbot,ztop)
        ax[0].xaxis.tick_top()
        ax[0].set_xlabel('Vsh', color='.5')
        ax[0].xaxis.set_label_position('top')
        plt.setp(ax[0].xaxis.get_majorticklabels(), fontsize=8, color='.5')
        ax[0].grid()
    ax[1].plot(WW.VP, z, 'k', lw=linethickness, label='Vp')
    if flagvs:
        ax[1].plot(WW.VS, z, color='.7', lw=linethickness, label='Vs')
    ax[2].plot(WW.RHO, z, 'k', lw=linethickness)
    ax[3].plot(WW.IP, z, 'k', lw=linethickness)
    if flagvs:
        if 'PR' in WW.columns:
            rmin=WW.PR[(z>=ztop) & (z<=zbot)].min()
            rmax=WW.PR[(z>=ztop) & (z<=zbot)].max()
            ax[4].plot(WW.PR, z, 'k', lw=linethickness)
            ax[4].set_xlabel('PR')
        else:
            rmin=WW.VPVS[(z>=ztop) & (z<=zbot)].min()
            rmax=WW.VPVS[(z>=ztop) & (z<=zbot)].max()
            ax[4].plot(WW.VPVS, z, 'k', lw=linethickness)
            ax[4].set_xlabel('Vp/Vs')
    else:
        ax[4].text(2.5, (zbot-ztop)/2+ztop, 'no Vs', fontsize=14,horizontalalignment='center')
    ax[4].set_xlim(rmin-rmin*.01,rmax+rmax*.01)
    ax[1].set_xlabel('Velocity [m/s]'), ax[1].set_xlim(velmin-velmin*.1,velmax+velmax*.1)
    ax[1].legend(fontsize=8, loc='lower right')
    ax[2].set_xlabel('Density [g/cc]'), ax[2].set_xlim(dmin-dmin*.01,dmax+dmax*.01)
    ax[3].set_xlabel('AI [m/s*g/cc]'),  ax[3].set_xlim(ipmin-ipmin*.1,ipmax+ipmax*.1)
    for i,aa in enumerate(ax):
        if i!=0:
            aa.grid()
            plt.setp(aa.xaxis.get_majorticklabels(), fontsize=8)
        if flagtops: # plot top markers on all columns
            for topn,topz in tops.iteritems():
                if (topz>=ztop) & (topz<=zbot):
                    aa.axhline(y=topz,color=color_top,alpha=alpha_top)
                    if i==4:
                        aa.text(x=rmax+rmax*.01,y=topz,s=topn,**format_tops)
    if name is not None:
        plt.suptitle(name, **format_title_topright)
    plt.subplots_adjust(hspace=.05,top=.9,bottom=.1,left=.08,right=.98)

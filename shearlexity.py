# Shearlexity: spatial complexity-entropy spectra via shearlets

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from FFST import (scalesShearsAndSpectra,
                  inverseShearletTransformSpect,
                  shearletTransformSpect)
from FFST._fft import ifftnc  # centered nD inverse FFT

logfn = np.log2

def map_cecp(img, startscale=4, rho=3,pad_to=0, Psi=None, with_plots=True):
    """
    Calculate and show Comlexity-Entropy spatial maps for an input image
    """

    img,npad = pad_img(img, pad_to)
    nr,nc = img.shape
    crop = (slice(npad,-npad),)*2
    H,C = local_cecp(img,startscale,rho=rho,Psi=Psi)
    if npad > 0:
        img,H,C = (a[crop] for a in (img, H,C))
    
    if with_plots:
        to_show = (img, H,C )
        f, axs = plt.subplots(1,3,figsize=(12,4))
        titles = ('input','H','C','mask')
        cmaps = ('gray', 'winter','summer')
        for ax, img, title,cm in zip(axs, to_show, titles,cmaps):
            imh = ax.imshow(img,cmap=cm)
            ax.set_title(title)
            plt.colorbar(mappable=imh, ax=ax)

        plt.tight_layout()

        f.set_size_inches(f.get_size_inches()+np.array([2.5,0]))
    return H,C


def local_cecp(img,startscale=2,rho=3,Psi=None):
    "cecp-shearlets: complexity-entropy causality pair for an image (local)"

    img = parity_crop(img)
    img_range = np.abs(img.max()-img.min())
    ST,_ = shearletTransformSpect(img,Psi=Psi)
    
    _,_,ncomps = ST.shape
    slx = scale_slices(ncomps)
    
    scale_starts = [sl.start for sl in slx]
    kstart = scale_starts[startscale]

    details = ST[...,slx[-1]]
    sdmap = np.std(details,-1)
    details_range = abs(details.max()-details.min())
    
    ST = ST[...,kstart:]
    ST = ST**2
    
    nscales = len(slx[startscale:])

    # equalize band energies
    ax = 1/(4**np.arange(len(slx)+2))
    for i, sl in enumerate(slx[startscale:]):
        sl2 = slice(sl.start-kstart,sl.stop-kstart)
        ST[...,sl2] *= ax[i]**(6/4)
    ST[...,slx[-1]] *= 2.5 # with this the energy spectrum is more uniform
    
    if rho > 0:
        for i,sl in enumerate(slx[startscale:]):
            sl2 = slice(sl.start-kstart,sl.stop-kstart)
            rhox = rho*2**(nscales-i-1)
            ST[...,sl2] = ndimage.gaussian_filter(ST[...,sl2],sigma=(rhox,rhox,0))
    
    probs = ST/ST.sum(-1)[:,:,None] 
    
    S = -np.sum(probs*logfn(1e-8+probs),-1)
    M = probs.shape[-1]
    PPr = 0.5*(probs+1/M)
    J = -np.sum((PPr)*logfn(1e-8+PPr),-1) - 0.5*(S + logfn(M))
    Hr = S/logfn(M)
    Cr = J*Hr/jmax(M)
    
    return Hr, Cr

def global_cecp(ST,startscale=2, central_slice=None,):
    "cecp-shearlets: complexity-entropy causality pair for an image (global)"
    nr,nc,fullncomps = ST.shape
    slx = scale_slices(fullncomps)
    ST = ST**2    
    scale_starts = [sl.start for sl in slx]
    kstart = scale_starts[startscale]
    if central_slice is None:
        central_slice = (slice(None),slice(None))
        
    E = ST[central_slice].sum(0).sum(0) # sum over all locations, analogous to Ej in Rosso et al 2001
    E = rescale_edsv(E)
    E = E[kstart:]
    prob = E/np.sum(E)
    
    return cecp(prob)



# ------ Auxiliary functions ---------------

def jmax(M):
    "Max Jensen-Shannon divergence in a system of M states"
    return -0.5*((M+1)*logfn(M+1)/M -2*logfn(2*M) + logfn(M))


def parity_crop(f):
    "Crop image if not both sides have either even or odd number of pixels"
    sh = f.shape
    parity = [s%2==0 for s in sh]
    if parity[0] != parity[1]:
        k = np.argmax(sh)
        if k == 0:
            f = f[1:,:]
        else:
            f = f[:,k:]
    return f


def scale_slices(N):
    "Sub-indices for each spatial scale"
    out = [slice(0,1)]
    n = 1
    k = 2
    while n < N:
        out.append(slice(n,n+2**k))
        n = n+2**k
        k += 1
    return out


def shannon(P):
    "Shannon entropy for distribution {P}"
    return -np.sum(P[P>0]*logfn(P[P>0]))
 

def cecp(P):
    "cecp: calculate complexity-entropy causality pair for distribution {P}"
    M = len(P)
    Pe = np.ones(M)/M
    Sp = shannon(P)
    Smax = logfn(M)
    J = shannon(0.5*(P+Pe)) - 0.5*(Sp + Smax)
    Hr = Sp/Smax
    Cr = (J/jmax(M))*Hr
    return (Hr,Cr)

def rescale_edsv(E):
    "equalize power of shearlet coefficients for random images"
    slx = scale_slices(len(E))
    ax = 1/(4**np.arange(len(slx)+2))
    Ex=np.zeros_like(E)
    for i, sl in enumerate(slx):
        Ex[sl] = E[sl]*ax[i]**(6/4)
    Ex[slx[-1]] *= 2.5 # this make energy spectrum uniform
    return Ex
   
def rescale_EST(ST,startscale=0):
    "equalize power of shearlet coefficients for random images"    
    slx = scale_slices(ST.shape[-1])
    ax = 1/(4**np.arange(len(slx)+2))
    scale_starts = [sl.start for sl in slx]
    kstart = scale_starts[startscale]
    for i, sl in enumerate(slx[starscale:]):
        sl2 = slice(sl.start-kstart,sl.stop-kstart)
        ST[...,sl2] *= ax[i]**(3/2)
    ST[...,slx[-1]] *= 2.5 
    return ST

def pad_img(img,targ=1024):
    npad = max(0,int(np.ceil(np.max(targ-np.array(img.shape))*0.5)))
    return np.pad(img, npad, 'constant'),npad
        

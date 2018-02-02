import numpy as np
from numba import jit, autojit,prange

from numpy.random import randint, uniform, rand


#@autojit
@jit(nogil=True,parallel=True)
def mcstep(m,beta):
    if np.ndim(beta) < 2:
        beta = beta*ones(m.shape, dtype=float32)
    nr,nc = m.shape
    nsweeps = nr*nc
    rr,cc = randint(nr,size=nsweeps),randint(nc,size=nsweeps)
    for sweep in prange(nsweeps):
        r,c = rr[sweep],cc[sweep]
        s = m[r,c]

        E = m[(r+1)%nr,c] + m[(r-1)%nr,c] + m[r,(c+1)%nc] + m[r,(c-1)%nc]
        E *= 2.*s
        if E < 0 or (rand() < np.exp(-E*beta[r,c])):
            m[r,c] *= -1                   
    return m

Tc = 2.0/(np.log(1.0 + 2**0.5))

def ising_surf(Tr,m=None,shape=(256,256),burnin=1e4,niters=1e5):
    if m is None:
        #m = np.where(uniform(size=shape)<0.5,-1,1)
        m = -np.ones(shape)
    out = np.zeros(m.shape)
    beta = 1./(Tr*Tc)
    for i in range(int(niters+burnin)):
        m = mcstep(m,beta)
        if i >= burnin:
            out += m
    return out/niters

def make_state(shape=(256,256)):
    return np.where(uniform(size=shape)<0.5,-1,1).astype(int)

def ising_surf2(T,shape=(256,256),burnin=1000,niters=1e4):
    m = make_state(shape)
    acc = []
    #acc.append(m.copy())
    out = np.zeros(m.shape)
    beta = 1./T
    
    for i in range(int(niters+burnin)):
        #z =  np.zeros_like(m)
        m = mcstep(m,beta)
        acc.append(abs(np.mean(m)))
        #x = mcstep2(m,z,beta)
        if i >= burnin:
            out += m
        #if not i%1000:
        #    acc.append(m.copy())
    #acc.append(m.copy())
    return out/niters,acc#/niters

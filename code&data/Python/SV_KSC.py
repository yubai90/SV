# -*- coding: utf-8 -*-
"""
This is a script file to replicate the basic SV model in KSC(1998).
Reference: Kim, S., Shepherd, N., and Chib, S. (1998), “Stochastic Volatility: 
Likelihood Inference and Comparison with ARCH models,” Review of Economic Studies,
65, 361–393. 

@author: YU BAI

"""
import numpy as np
import scipy as sci
import os
os.chdir('C:\\Users\\3029133\\Documents\\Graziani\\Final')
from pandas import read_csv
from pandas import DataFrame
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


## sampling s and h

def SVRW(ystar, h, h0, sigh2):
    T=len(h)
    # normal mixture
    pj=np.array([0.0073,0.10556,0.00002,0.04395,0.34001,0.24566,0.2575]) 
    mj=np.array([-10.12999,-3.97281,-8.56686,2.77786,0.61942,1.79518,-1.08819])-1.2704 # means are adjusted
    sigj2=np.array([5.79596,2.61369,5.17950,0.16735,0.64009,0.34023,1.26261])
    sigj=np.sqrt(sigj2)
    
    # sample s from a 7-point discrete distribution
    temprand=np.random.uniform(0,1,T)
    q=np.array([pj,]*T)+norm.pdf(np.array([ystar,]*7).T,np.array([h,]*7).T+np.array([mj,]*T),np.array([sigj,]*T))
    q=q/np.array([np.sum(q,axis=1),]*7).T
    s=7-np.sum(np.array([temprand,]*7).T<np.cumsum(q,axis=1),axis=1)+1
    
    # sample h
    H=sci.sparse.eye(T).toarray()-sci.sparse.eye(T,k=-1).toarray()
    HH=np.asarray(np.asmatrix(H).T*np.asmatrix(H))
    d_s=np.asarray(np.asmatrix(mj[np.asarray(s-1)]).T)
    diagonals=1/sigj2[np.asarray(s-1)]
    iSig_s=sci.sparse.diags(diagonals).toarray()
    Kh=HH/sigh2+iSig_s
    Kh=np.asmatrix(Kh)
    h_hat=np.linalg.inv(Kh)*(h0/sigh2*np.asmatrix(HH)*np.asmatrix(np.ones((T,1)))+np.asmatrix(iSig_s)*np.asmatrix((ystar.reshape(T,1)-d_s)))
    chol_Kh=np.linalg.inv(np.linalg.cholesky(np.asmatrix(Kh)).T)*np.asmatrix(np.random.normal(0,1,T)).T
    h=h_hat+np.asarray(chol_Kh)
    h=np.asarray(h).reshape(T)
    return h

# load the data
nsim=10000
burnin=500
y=read_csv('CPNUSD.csv',header=None)
y=y.values
T=len(y)

# prior
mu0=0
Vmu=100
a0=0
b0=100
nu_h=3
S_h=0.2*(nu_h-1)

# initialize the Markov chain
sigh2=0.05
mu=np.mean(y)
h0=np.log(np.var(y))
h=h0*np.ones((T,1)).reshape(T)
H=sci.sparse.eye(T).toarray()-sci.sparse.eye(T,k=-1).toarray()
HH=np.asarray(np.asmatrix(H).T*np.asmatrix(H))

# initialize for storage 
store_theta=np.zeros((nsim,3))
store_h=np.zeros((nsim,T))

# MCMC step
for isim in range(nsim+burnin):
    # sample mu
    Kmu=1/Vmu+np.sum(1/np.exp(h))
    mu_hat=(mu0/Vmu+np.sum(y.reshape(T)/np.exp(h)))/Kmu
    mu=mu_hat+np.random.normal(0,1)/np.sqrt(Kmu)
    
    # sample h
    ystar=np.asarray(np.log(np.square(y-mu)+0.0001)).reshape(T)
    h=SVRW(ystar,h,h0,sigh2)
    
    # sample sigh2
    sigh2=1/np.random.gamma(nu_h+T/2,1/(S_h+np.asmatrix(h-h0)*np.asmatrix(HH)*np.asmatrix(h-h0).T/2))
    sigh2=sigh2[0,0]
    
    # sample h0
    Kh0=1/b0+1/sigh2
    h0_hat=(a0/b0+h[0]/sigh2)/Kh0
    h0=h0_hat+np.random.normal(0,1)/np.sqrt(Kh0)
    
    if isim+1>burnin:
        isave=isim-burnin
        store_h[isave,:]=h.reshape(1,T)
        store_theta[isave,:]=np.array([mu,h0,sigh2])
        
        
theta_hat=np.mean(store_theta,0)
df=DataFrame(store_theta)
theta_CI=df.quantile([0.025,0.975])
h_hat=np.mean(np.exp(store_h/2),0).reshape(T,1) 

# graph plot
t = np.linspace(2005,2013,T)
params={'figure.figsize'   : '20, 5'}    # set figure size
pylab.rcParams.update(params)
plt.plot(t,h_hat,'b')
        
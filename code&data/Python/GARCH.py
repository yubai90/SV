# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:44:14 2017
This script is for estimation of GARCH model.

Reference: Bollerslev, T. (1986) “Generalized Autoregressive Conditional 
Heteroskedasticity”, Journal of Econometrics, 31, 307-327. 

@author: YU BAI
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import size, log, pi, sum, array, zeros, diag, dot, mat, asarray, sqrt,copy
from numpy.linalg import inv
from scipy.optimize import fmin_slsqp
from pandas import read_csv
import os
os.chdir('C:\\Users\\3029133\\Documents\\Graziani\\Final')

def garch_likelihood(parameters, data, sigma2,out=None):
    ''' Returns likelihood for GARCH(1,1) model.'''
    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    beta = parameters[3]

    T = size(data,0)
    eps = data - mu
    # Data and sigma2 are T by 1 vectors
    for t in range(1,T):
        sigma2[t] = (omega + alpha * eps[t-1]**2 + beta * sigma2[t-1])

    logliks = 0.5*(log(2*pi) + log(sigma2) + eps**2/sigma2)
    loglik = sum(logliks)

    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma2)


def constraint(parameters, data, sigma2, out=None):
    ''' Constraint that alpha+beta<=1'''

    alpha = parameters[2]
    beta = parameters[3]

    return array([1-alpha-beta])


def hessian_2sided(fun, theta, args):
    f = fun(theta, *args)
    h = 1e-5*np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = size(theta,0)
    h = np.diag(h)

    fp = zeros(K)
    fm = zeros(K)
    for i in range(K):
        fp[i] = fun(theta+h[i], *args)
        fm[i] = fun(theta-h[i], *args)

    fpp = zeros((K,K))
    fmm = zeros((K,K))
    for i in range(K):
        for j in range(i,K):
            fpp[i,j] = fun(theta + h[i] + h[j],  *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(theta - h[i] - h[j],  *args)
            fmm[j,i] = fmm[i,j]

    hh = (diag(h))
    hh = hh.reshape((K,1))
    hh = dot(hh,hh.T)

    H = zeros((K,K))
    for i in range(K):
        for j in range(i,K):
            H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f
                       + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
            H[j,i] = H[i,j]

    return H


# Import data
FTSEreturn = read_csv('CPNUSD.csv')
FTSEreturn=FTSEreturn.values
T=len(FTSEreturn)
FTSEreturn=FTSEreturn.reshape(T)


# Starting values
startingVals = array([FTSEreturn.mean(),
                      FTSEreturn.var() * .01,
                      .03, .90])

# Estimate parameters
finfo = np.finfo(np.float64)
bounds = [(10*FTSEreturn.mean(), -10*FTSEreturn.mean()),
          (finfo.eps, 2*FTSEreturn.var() ),
          (0.0,1.0), (0.0,1.0)]

T = size(FTSEreturn,0)
sigma2 = np.repeat(FTSEreturn.var(),T)
args = (FTSEreturn, sigma2)
estimates = fmin_slsqp(garch_likelihood, startingVals, \
           f_ieqcons=constraint, bounds = bounds, \
           args = args)

loglik, logliks, sigma2final = garch_likelihood(estimates, \
                               FTSEreturn, sigma2, out=True)


step = 1e-5 * estimates
scores = np.zeros((T,4))
for i in range(4):
    h = step[i]
    delta = np.zeros(4)
    delta[i] = h

    loglik, logliksplus, sigma2 = garch_likelihood(estimates + delta, FTSEreturn, sigma2, out=True)
    loglik, logliksminus, sigma2 = garch_likelihood(estimates - delta, FTSEreturn, sigma2, out=True)
    scores[:,i] = (logliksplus - logliksminus)/(2*h)

I = np.dot(scores.T,scores)/T


J = hessian_2sided(garch_likelihood, estimates, args)
J = J/T
Jinv = mat(inv(J))
vcv = Jinv*mat(I)*Jinv/T
vcv = asarray(vcv)


output = np.vstack((estimates,sqrt(diag(vcv)),estimates/sqrt(diag(vcv)))).T
print('Parameter   Estimate       Std. Err.      T-stat')
param = ['mu','omega','alpha','beta']
for i in range(len(param)):
    print('{0:<11} {1:>0.6f}        {2:0.6f}    {3: 0.5f}'.format(param[i],output[i,0],output[i,1],output[i,2]))

# Produce a plot
dates = np.linspace(2005,2013,T)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dates,np.sqrt(252*sigma2))
fig.autofmt_xdate()
ax.set_ylabel('Volatility')
ax.set_title('Volatility (GARCH(1,1))')
plt.show()


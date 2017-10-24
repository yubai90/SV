# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:58:36 2017

@author: YU BAI
"""

import numpy as np
from numpy import log,sqrt
from scipy.stats import chi2
from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
df = 1

x = np.linspace(0,
               10, 50)
ax.plot(x, log(chi2.pdf(x, df)/norm.pdf(x,-1.2704,sqrt(4.93))),
         'b-', lw=5, alpha=0.6, label='chi2 pdf')
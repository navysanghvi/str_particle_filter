# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:17:29 2016

@author: Jimit Gandhi
"""

import numpy as np
import py
import numpy.random
from numpy.random import multinomial

def get_p_upd(self,wt_vect,X_upt)
    wt_vect = wt_vect/np.sum(wt_vect)
    wt_vect_reshaped = np.reshape(wt_vect,len(wt_vect))
    distribution = np.random.multinomial(len(wt_vect),wt_vect,1)
    X_new = np.empty([1,3])
    for particle in range(len(distribution[0])):
        X_new = np.concatenate((X_new,np.tile(X_upd[particle],(distribution[0,particle],1))))
    X_new = X_new[1:len(X_new),:]
    return X_new
    
from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython 
@cython.boundscheck(False)
@cython.wraparound(False)

def thresholding3_cy(np.ndarray[DTYPE_t, ndim=1] x=None, np.ndarray[DTYPE_t, ndim=1] a=None, np.ndarray[DTYPE_t, ndim=1] t=None, 
                    DTYPE_t thr=0.0, DTYPE_t thr_d=0.0, int dl=1):
    cdef int x_length = x.shape[0]
    cdef int i = 0
    cdef list t_start = []
    cdef list i_start = []
    cdef list a_start = []
    cdef list t_end = []
    cdef list i_end = []
    cdef list a_end = []
    
    while i<x_length:
        while i<x_length and x[i]>=thr:
            i+=1
        if a[i-dl:i].shape[0]!=0:
            i_start.append(i-dl+np.argmin(a[i-dl:i]))
            t_start.append(t[i-dl+np.argmin(a[i-dl:i])])
            a_start.append(np.min(a[i-dl:i]))
        while i<x_length and x[i]<thr_d:
            i+=1
        if a[i:i+dl].shape[0]!=0 and len(i_start)-len(i_end)==1:
            i_end.append(i+np.argmax(a[i:i+dl]))
            t_end.append(t[i+np.argmax(a[i:i+dl])])
            a_end.append(np.max(a[i:i+dl]))
        i+=1
    return i_start, i_end, t_start, t_end, a_start, a_end
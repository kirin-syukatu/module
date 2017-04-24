from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython 
@cython.boundscheck(False)
@cython.wraparound(False)
def moving_average_cy(np.int n=None, np.ndarray[DTYPE_t, ndim=1] x=None):
    assert n%2==1, '\n{0}\naveraging number must be odd\n{1}'.format('_'*80, '-'*80)
    cdef int x_length = x.shape[0]
    cdef int m, i, j
    cdef DTYPE_t _av
    cdef np.ndarray[DTYPE_t, ndim=1] x_ave = np.zeros(x_length, dtype=DTYPE)

    m=(n-1)//2
    npm=np.mean
    _av=np.mean(x[0:m+1]) #m+1 ko -> 2m+1 ko
    for i in range(m):
        x_ave[i]=_av
        _av *= (i+m+1) / (i+m+2)
        _av += x[i+m+1] / (i+m+2)

    _av=np.mean(x[0:m*2+1]) #2m+1 ko -> 2m+1 ko
    for i in range(m, x.shape[0]-m-1):
        x_ave[i]=_av
        _av -= x[i-m]/(2*m+1)
        _av += x[i+m+1]/(2*m+1)

    _av=np.mean(x[x.shape[0]-(m+m+1):x.shape[0]]) #2m+1 ko -> m+1 ko
    for j, i in enumerate(range(x.shape[0]-(m+1), x.shape[0])):
        x_ave[i]=_av
        _av *= (2*m+1-j) / (2*m-j)
        _av -= x[i-m] / (2*m-j)

    return x_ave

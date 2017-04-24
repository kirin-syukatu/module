import time, os
from functools import wraps
import numpy as np
print("my_module"+os.sep+"time_deco.py is loaded")
global TIME_PRINT
TIME_PRINT = 0

proc_num=-1
def time_deco(_a, cn):
    def wrappedwrapper(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            global proc_num
            proc_num+=1
            now_num=proc_num
            if _a:
                print( '\n{0:0>3}:{1}.{2}({3}, {4}) {5}'.format(proc_num, cn, str(f.__name__), 
                    [i if not isinstance(i, list) else np.array(i) for j,i in enumerate(args) if not j==0], kwargs, '-'*20))
                with open('log.txt', 'a') as fl:
                    fl.write('\n\r{0:0>3}:{1}.{2}({3}, {4}) {5}\n\r'.format(proc_num, cn, str(f.__name__), 
                        [i if not isinstance(i, list) else np.array(i) for j,i in enumerate(args) if not j==0], kwargs, '-'*20))

            before = time.time()
            result = f(*args, **kwargs)
            after = time.time()
            if _a:
                print('{0}{1:0>3} was used {2} sec\n'.format('-'*20, now_num, float(after - before))) 
                with open('log.txt', 'a') as fl:
                    fl.write('\n\r{0}{1:0>3} was used {2} sec\n\r'.format('-'*20, now_num, float(after - before)))
            return result            
        return wrapper
    return wrappedwrapper

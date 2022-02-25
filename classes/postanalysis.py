import numpy as np
import multiprocessing as mp
import time
# shared parameters
default_num_threads = 12


def parallel_reduce_matrix(data_list, function='nanmedian', axis=0, num_threads = default_num_threads, verbose=False):
    """Parallel processing of function to reduce dimension, """
    _start_time = time.time()
    _n_jobs = len(data_list[0])
    _func = getattr(np, function)
    _args = []

    for _i in np.arange(_n_jobs):
        _args.append(
            (np.array([_data[_i] for _data in data_list]), axis) # axis=0
        )
    
    with mp.Pool(num_threads) as reduce_pool:
        results = reduce_pool.starmap(_func, _args, chunksize=1)
        reduce_pool.close()
        reduce_pool.join()
        reduce_pool.terminate()
    if verbose:
        print(f"{len(data_list)} processed in {time.time()-_start_time:.3f}s. ")

    return np.array(results)

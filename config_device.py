from __future__ import print_function
import os
import tensorflow as tf

def config_device(arch):

    if arch == 'default':
        return None

    # common
    os.environ["KMP_BLOCKTIME"] = "1"
    #os.environ["KMP_SETTINGS"] = "1"
    # os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    os.environ["KMP_AFFINITY"]= "granularity=fine,compact,1,0"

    #arch-specific stuff
    if arch=='HSW':
        num_inter_threads = 2
        num_intra_threads = 16
    elif arch=='KNL':
        num_inter_threads = 2
        num_intra_threads = 32
    else:
        raise ValueError('Please specify a valid architecture with arch (allowed values: HSW, KNL)')

    #set the rest
    os.environ['OMP_NUM_THREADS'] = str(num_intra_threads)
    print("Using ",num_inter_threads,"-way task parallelism with ",num_intra_threads,"-way data parallelism.")

    return tf.ConfigProto(inter_op_parallelism_threads=num_inter_threads,intra_op_parallelism_threads=num_intra_threads)

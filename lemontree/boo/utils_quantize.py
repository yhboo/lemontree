import theano
import theano.tensor as T
import numpy as np

def quantize(input, step_size, n_bits):
    '''
    this function quantizes input using step_size, n_bits
    input : any tensor type(maybe)
        variable which will be quantized
    step_size : float scalar
        decides step
    n_bits : positive integer scalar
        decides precision

    return : quantized_input (step_size multiplied)
    '''
    #print('in_quntize')
    #print('w : ',input)
    #print('s : ',step_size)
    #print('n : ',n_bits)
    precision = T.pow(2, n_bits-1)-1

    return T.cast(T.maximum(T.minimum(T.round(input / step_size), precision), -precision) * step_size, dtype = theano.config.floatX)



def get_step_size(x, n_bits, threshold = 0.000001, initial_step_size = 0.00001):
    '''
    x : numpy array
    n_bits : positive integer

    '''
    mean = np.mean(x)
    std = np.std(x)

    M = 2**(n_bits) - 1

    step_size = initial_step_size
    prev_step_size = step_size

    for i in range(100):
        z = np.sign(x) * np.minimum(np.round((np.abs(x) / step_size)), (M-1)/2)
        step_size = np.sum(x*z) / np.sum(z**2)
        if np.abs(prev_step_size - step_size) < threshold:
            break
        else:
            prev_step_size = step_size

    return np.asarray(step_size, dtype = theano.config.floatX)

        

'''
-added by yhboo-
dense layer with fixed point weight
enable fixed weight propagation
'''


import numpy as np
import theano
import theano.tensor as T
from lemontree.initializers import *
from collections import OrderedDict
from baselayer import BaseLayer

from utils_quantize import *


class FixedDenseLayer(BaseLayer):
    '''
    This class implements fixed dense layer
    '''

    def __init__(self, input_shape, output_shape,
                 n_bits_w, step_size_w = None,
                 n_bits_b = 1, step_size_b = None,
                 use_bias = True, fix_weight_only = True,
                 W = None, b = None, initializer = Uniform(),
                 name = None):
        """
        This function initializes the class.
        Input is 2D tensor, output is 2D tensor.
        For efficient following batch normalization, use_bias = False.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
        output_shape: tuple
            a tuple of single value, i.e., (output dim,)
        n_bits_w, n_bits_b : integer scalar
            decides weights/bias precision
        step_size_w, step_size_b : float scalar
            decides weights/bias step
        use_bias: bool, default: True
            a bool value whether we use bias or not.
        fix_weight_only : bool, default: True
            a bool value whether we fix only weights or both weights & bias
        name: string
            a string name of this layer.

        Returns
        -------
        None.
        """

        super(FixedDenseLayer, self).__init__(name)

        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(use_bias, bool), '"use_bias" should be a bool value.'
        assert isinstance(n_bits_w, int) and n_bits_w >0, '"n_bits_w" should be a positive integer.'
        assert isinstance(n_bits_b, int) and n_bits_b >0, '"n_bits_b" should be a positive integer.'

        #assert (use_bias == False and fix_weight_only == False and step_size_b == None) or (use_bias == True and fix_weight_only == True and step_size_b == None), 'no bias to quantize'
        #assert isinstance(step_size_w, theano.config.floatX), '"step_size_w" should be '+theano.config.floatX
        #assert isinstance(step_size_b, theano.config.floatX), '"step_size_w" should be '+theano.config.floatX



        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias
        self.fix_weight_only = fix_weight_only


        # create shared variables
        """
        Shared Variables
        ----------------
        W: 2D matrix
            shape is (input dim, output dim).
        b: 1D vector
            shape is (output dim,).
        W_fixed : 2D matrix
            shape is (input dim, output dim).
            all values are quantized
        b_fixed : 1D vector
            shape is (output dim,).
            all values are quantized if fix_weight_only = False

        """


        if W:
            Weights = W
        else:
            Weights = initializer((input_shape[0], output_shape[0]))

        if b:
            bias = b
        else:
            bias = np.zeros(output_shape).astype(theano.config.floatX)

        if step_size_w:
            s_w = step_size_w
        else:
            s_w = get_step_size(Weights, n_bits_w)

        if step_size_b:
            s_b = step_size_b
        else:
            s_b = get_step_size(bias, n_bits_b)
           


        
        #self.W = theano.shared(Weights, self.name + '_weight')
        self.W = theano.shared(np.asarray(Weights, dtype = theano.config.floatX), name = 'temp')
        self.W.tags = ['weight', self.name]
        
        self.b = theano.shared(np.asarray(bias, dtype = theano.config.floatX))
        self.b.tags = ['bias', self.name]

        self.W_fixed = theano.shared(np.asarray(Weights, dtype = theano.config.floatX))
        self.W_fixed.tags = ['weight_fixed', self.name]

        self.b_fixed = theano.shared(np.asarray(bias, dtype = theano.config.floatX))
        self.b_fixed.tags = ['bias_fixed',self.name]

        self.n_bits_w = theano.shared(n_bits_w)
        self.n_bits_b = theano.shared(n_bits_b)

        self.step_size_w = theano.shared(s_w, allow_downcast = True)
        self.step_size_b = theano.shared(s_b, allow_downcast = True)

        self.fixed_updates = []


    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        
        Math Expression
        -------------------
        Y = dot(X, W) + b
        Y = dot(X, W)
            bias is automatically broadcasted. (supported theano feature)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        if self.use_bias:
            if self.fix_weight_only == True:
                return T.dot(input_, self.W_fixed) + self.b
            else:
                return T.dot(input_, self.W_fixed) + self.b_fixed
        else:
            return T.dot(input_, self.W_fixed)

    def get_params_fixed(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        list
            a list of shared variables used.
        """
        if self.use_bias:
            if self.fix_weight_only == True:
                return [self.W_fixed, self.b]
            else:
                return [self.W_fixed, self.b_fixed]
        else:
            return [self.W_fixed]    
    def get_params_float(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        list
            a list of shared variables used.
        """
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    def get_fixed_updates(self):
        
        self.fixed_updates.append((self.W_fixed, quantize(self.W, self.step_size_w, self.n_bits_w)))
        if self.fix_weight_only == False:
            self.fixed_updates.append((self.b_fixed, quantize(self.b_fixed, self.step_size_b, self.n_bits_b)))
        '''
        precision = T.cast(T.pow(2, (self.n_bits_w-1))-1, theano.config.floatX)
        self.fixed_updates.append((self.W_fixed, T.cast(T.maximum(T.minimum(T.round(self.W / self.step_size_w), precision), -precision) * step_size_w, dtype = 'float32')))
        if self.fix_weight_only == False:
            precision = T.pow(2, (self.n_bits_b-1))-T.cast(1,int)
            self.fixed_updates.append((self.b_fixed, T.cast(T.maximum(T.minimum(T.round(self.b / self.step_size_b), precision), -precision) * step_size_b, dtype = 'float32')))
        '''


        return OrderedDict(self.fixed_updates)
        
 
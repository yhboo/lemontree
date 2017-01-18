import numpy as np
import theano
import theano.tensor as T

from baselayer import BaseLayer 



class ReLU(BaseLayer):
    """
    This class implements ReLU activation function.
    """
    def __init__(self, alpha=0, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        alpha: float, default: 0
            a positive float value which indicates the tangent of x < 0 range.
            if alpha is not 0, this function become a leaky ReLU.
        name: string
            a string name of this class.

        Returns
        -------
        None.
        """
        super(ReLU, self).__init__(name)
        # check asserts
        assert alpha >= 0, '"alpha" should be a non-negative float value.'

        # set members
        self.alpha = alpha        

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        ReLU is element-wise operation.

        Math Expression
        -------------------
        y = maximum(x, 0)
        y = ifelse(x > 0, x, \alpha * x)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.relu(input_, self.alpha)



class Softmax(BaseLayer):
    """
    This class implements softmax activation function.
    """
    def __init__(self, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        name: string
            a string name of this class.

        Returns
        -------
        None.
        """
        super(Softmax, self).__init__(name)

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Softmax converts output energy to probability distributuion.

        Math Expression
        -------------------
        y_k = exp(x_k) / \sum(exp(x_i))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.softmax(input_)
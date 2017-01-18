import numpy as np
import h5py
import scipy.io


class parameters_h5(object):

    def __init__(self, path):
        self.weight_all = h5py.File(path,'r')
        print(self.weight_all.keys())
#        print self.weight_all.get('timedistributed_1').keys()


    def load_param(self, paramType = 'W', path = 1, modelType = 'dense', cellType = ''):
        '''
        paramType : dense, convolution2d, timedistributed -> 'W', 'b'
                    lstm -> 'U','W','b'
                    batchnormalization -> 'beta', 'gamma', 'running_mean', 'running_std'

        modelType : 'dense', 'convolution2d', 'timedistributed', 'lstm', 'batchnormalization'

        cellType  : dense, convolution2d, timedistributed -> not used
                    lstm -> 'c', 'f', 'i', 'o'
        '''    

        if modelType == 'lstm':
            name = modelType+'_'+str(path)+'/'+modelType+'_'+str(path)+'_'+paramType+'_'+cellType
        
        elif modelType == 'timedistributed':
            name = modelType+'_'+str(path)+'/dense_'+str(path)+'_'+paramType

        else:
            name = modelType+'_'+str(path)+'/'+modelType+'_'+str(path)+'_'+paramType
        
        param = self.weight_all.get(name)
        
        assert(param != None), 'wrong modeltype while load_param'

        return param

    def show_list(self, path=None):
        if path==None:
            print(self.weight_all.keys())
        else:
            print(self.weight_all.get(path).keys())



if __name__ == '__main__':
    a = parameters_h5('D:/boo/theano_data/models/mnist_final.h5')

    

    a.show_list()
    b = a.weight_all.get('dense_1/dense_1_W')

    print(b.shape)
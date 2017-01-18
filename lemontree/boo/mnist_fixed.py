import numpy as np
import theano
import theano.tensor as T
import time

from lemontree.data.mnist import MNIST
from lemontree.data.generators import SimpleGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.initializers import *
from lemontree.objectives import CategoricalAccuracy, CategoricalCrossentropy
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params
from lemontree.utils.type_utils import merge_dicts
from lemontree.utils.graph_utils import get_inputs_of_variables

from graph import SimpleGraph
from layers import FixedDenseLayer
from optimizers import Adam
from weight_processing import *
from activation import ReLU, Softmax



np.random.seed(9999)
base_path = 'D:/boo/theano_data/'
experiment_name = 'mnist_mlp_fixed'

data_path = base_path + 'data/'
model_path = base_path + 'models/mnist_final_20.h5'

n_bits_w_all = 2
#================Prepare data================#

mnist = MNIST(data_path, 'flat')
mnist.split_train_valid(50000)
train_data, train_label = mnist.get_fullbatch_train()
test_data, test_label = mnist.get_fullbatch_test()
valid_data, valid_label = mnist.get_fullbatch_valid()

train_gen = SimpleGenerator('train', 250)
train_gen.initialize(train_data, train_label)
test_gen = SimpleGenerator('test', 250)
test_gen.initialize(test_data, test_label)
valid_gen = SimpleGenerator('valid', 250)
valid_gen.initialize(valid_data, valid_label)

#================Build graph================#

x = T.fmatrix('X')
y = T.ivector('y')



param_source = parameters_h5(model_path)


graph = SimpleGraph(experiment_name)
graph.add_input(x)
graph.add_layers([
    FixedDenseLayer(
        input_shape = (784,),
        output_shape = (20,),
        W = param_source.load_param('W', 1, 'dense'),
        b = param_source.load_param('b', 1, 'dense'),
        n_bits_w = n_bits_w_all,
        name='dense1'),
    ReLU(name='relu1'),
    
    FixedDenseLayer(
        input_shape = (20,),
        output_shape = (20,),
        W = param_source.load_param('W', 2, 'dense'),
        b = param_source.load_param('b', 2, 'dense'),
        n_bits_w = n_bits_w_all,
        name='dense2'),
    ReLU(name='relu2'),
    
    
    FixedDenseLayer(
        input_shape = (20,),
        output_shape = (10,),
        W = param_source.load_param('W', 3, 'dense'),
        b = param_source.load_param('b', 3, 'dense'),
        n_bits_w = n_bits_w_all,
        name='dense3'),
    Softmax(name='softmax1')])


loss = CategoricalCrossentropy().get_loss(graph.get_output(), y)
accuracy = CategoricalAccuracy().get_loss(graph.get_output(), y)



graph_params_fixed = graph.get_params_fixed()
graph_params_float = graph.get_params_float()
graph_updates = graph.get_updates()
graph_quantize = graph.get_fixed_updates()





#print('fixed')
#print(graph_params_fixed)
#print('float')
#print(graph_params_float)




#====================Prepare arguments===================#


optimizer = Adam(0.002)
optimizer_params = optimizer.get_params()
optimizer_updates = optimizer.get_updates(loss, graph_params_fixed, graph_params_float)

total_params = optimizer_params + graph_params_float
total_updates = merge_dicts([optimizer_updates, graph_updates])

############################
#have to add step size params
#Q : save optimizer params?
############################
params_saver = SimpleParameter(total_params, experiment_name + '_params/')
#params_saver.save_params()

lr_scheduler = LearningRateMultiplyScheduler(optimizer.lr, 0.2)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 5, 5)
hist.add_keys(['train_accuracy',  'valid_accuracy', 'test_accuracy'])

#================Compile functions================#

outputs = [loss, accuracy]
graph_inputs = get_inputs_of_variables(outputs)

train_func = theano.function(inputs=graph_inputs,
                             outputs=outputs,
                             updates=total_updates,
                             allow_input_downcast=True)

test_func = theano.function(inputs=graph_inputs,
                            outputs=outputs,
                            allow_input_downcast=True)

quantize_func = theano.function(inputs = [],
                                outputs = [],
                                updates = graph_quantize
                                )

#================Convenient functions================#

def train_trainset():
    graph.change_flag(1)
    train_loss = []
    train_accuracy = []
    for index in range(train_gen.max_index):
        trainset = train_gen.get_minibatch(index)
        train_batch_loss, train_batch_accuracy = train_func(*trainset)
        quantize_func()
        train_loss.append(train_batch_loss)
        train_accuracy.append(train_batch_accuracy)
    hist.history['train_loss'].append(np.mean(np.asarray(train_loss)))
    hist.history['train_accuracy'].append(np.mean(np.asarray(train_accuracy)))

def test_validset():
    graph.change_flag(-1)
    valid_loss = []
    valid_accuracy = []
    for index in range(valid_gen.max_index):
        validset = valid_gen.get_minibatch(index)
        valid_batch_loss, valid_batch_accuracy = test_func(*validset)
        valid_loss.append(valid_batch_loss)
        valid_accuracy.append(valid_batch_accuracy)
    hist.history['valid_loss'].append(np.mean(np.asarray(valid_loss)))
    hist.history['valid_accuracy'].append(np.mean(np.asarray(valid_accuracy)))

def test_testset():
    graph.change_flag(-1)
    test_loss = []
    test_accuracy = []
    for index in range(test_gen.max_index):
        testset = test_gen.get_minibatch(index)
        test_batch_loss, test_batch_accuracy = test_func(*testset)
        test_loss.append(test_batch_loss)
        test_accuracy.append(test_batch_accuracy)
    hist.history['test_loss'].append(np.mean(np.asarray(test_loss)))
    hist.history['test_accuracy'].append(np.mean(np.asarray(test_accuracy)))

#================Train================#
quantize_func()
graph.show_fixed_params()
test_testset()
print('direct quntization results')
hist.print_history_of_epoch()

change_lr = False
end_train = False
for epoch in range(50):
    if end_train:
        params_saver.load_params()
        break
    if change_lr:
        params_saver.load_params()
        lr_scheduler.change_learningrate(epoch)
        optimizer.reset_params()
    train_gen.shuffle()

    print('...Epoch', epoch)
    start_time = time.clock()

    test_testset()
    train_trainset()
    test_validset()

    end_time = time.clock()
    print('......time:', end_time - start_time)

    hist.print_history_of_epoch()
#    checker = hist.check_earlystopping()
    checker = 'keep_train'
    if checker == 'save_param':
        params_saver.save_params()
        change_lr = False
        end_train = False
    elif checker == 'change_lr':
        change_lr = True
        end_train = False
    elif checker == 'end_train':
        change_lr = False
        end_train = True
    elif checker == 'keep_train':
        change_lr = False
        end_train = False
    else:
        raise NotImplementedError('Not supported checker type')

#================Test================#

test_testset()
hist.print_history_of_epoch()
hist.save_history_to_csv()


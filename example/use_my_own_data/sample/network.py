# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx

def network():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=1500)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 1000)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 300)
    act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")
    fc4  = mx.symbol.FullyConnected(data = act3, name = 'fc4', num_hidden = 60)
    act4 = mx.symbol.Activation(data = fc4, name='relu4', act_type="relu")
    fc5  = mx.symbol.FullyConnected(data = act4, name='fc5', num_hidden=16)
    mlp  = mx.symbol.LinearRegressionOutput(data = fc5, label = label,name = 'softmax')
    return mlp





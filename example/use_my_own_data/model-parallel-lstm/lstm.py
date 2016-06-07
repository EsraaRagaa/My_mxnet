# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_label, dropout=0.):
    with mx.AttrScope(ctx_group='decode'):
        cls_weight = mx.sym.Variable("cls_weight")
        cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        with mx.AttrScope(ctx_group='layer%d' % i):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                              h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    loss_all = []
    for seqidx in range(seq_len):
        with mx.AttrScope(ctx_group='embed'):
            data = mx.sym.Variable("data/%d" % seqidx)

            # first conv
            conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=10)
            tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
            pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                                      kernel=(2,2), stride=(2,2))
            # second conv
            conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=10)
            tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
            pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                                      kernel=(2,2), stride=(2,2))
            # first fullc
            indata = mx.symbol.Flatten(data=pool2)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            with mx.AttrScope(ctx_group='layer%d' % i):
                next_state = lstm(num_hidden, indata=indata,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i, dropout=dp_ratio)
                indata = next_state.h
                last_states[i] = next_state
        # decoder
        if dropout > 0.:
            indata = mx.sym.Dropout(data=indata, p=dropout)
        with mx.AttrScope(ctx_group='decode'):
            label = mx.sym.Variable('label/%d' % seqidx)
            fc = mx.sym.FullyConnected(data=indata, weight=cls_weight, bias=cls_bias,
                                       num_hidden=num_label)
            sm = mx.sym.LogisticRegressionOutput(data=fc, label=label, name='t%d_sm' % seqidx)
            loss_all.append(sm)

    return mx.sym.Group(loss_all)




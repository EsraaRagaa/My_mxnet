# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import numpy as np
import mxnet as mx
import pyaudio
import struct
from bucket_io import BucketSentenceIter, default_build_vocab
import pylab as pl
import time
def Perplexity(label, pred):
    loss = 0.
    global num
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            loss += (pred[i][j] - label[i][j])*(pred[i][j] - label[i][j])
    return loss/pred.shape[0]

p = pyaudio.PyAudio()
stream = p.open(format = p.get_format_from_width(2),
                channels = 1,
                rate = 44100,
                output = True)

if __name__ == '__main__':
    batch_size = 1
    buckets = 30
    num_hidden = 2500
    num_label = 1500
    num_lstm_layer = 2
    num_epoch = 326

    print(batch_size, buckets, num_hidden, num_lstm_layer, num_epoch)

    img_data, wave_data = default_build_vocab("./data/data/2.mp4", "./data/data/2.mp3")

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    data_train = BucketSentenceIter(img_data,wave_data,
                                    buckets, batch_size, init_states, num_label)
    model = mx.model.FeedForward.load('model/lip', num_epoch, ctx=mx.context.gpu(1),
                                      num_epoch=500, learning_rate=0.5)
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    model.fit(X=data_train, eval_data=None,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),
              epoch_end_callback = mx.callback.do_checkpoint('model/lip'))




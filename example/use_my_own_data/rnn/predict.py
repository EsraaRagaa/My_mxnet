# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import numpy as np
import mxnet as mx
import pyaudio
import struct
from predict_io import BucketSentenceIter, default_build_vocab
import pylab as pl
import time

p = pyaudio.PyAudio()
stream = p.open(format = p.get_format_from_width(2),
                channels = 1,
                rate = 44100,
                output = True)

if __name__ == '__main__':
    batch_size = 1
    buckets = 30
    num_hidden = 2500
    num_label = 1445
    num_lstm_layer = 2
    num_epoch = 550

    print(batch_size, buckets, num_hidden, num_lstm_layer, num_epoch)

    img_data, wave_data = default_build_vocab("./data/4.mkv", "./data/4.mp3")

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    data_val = BucketSentenceIter(img_data, buckets, batch_size, init_states, num_label)
    model4 = mx.model.FeedForward.load('model/lip', num_epoch, ctx = mx.context.gpu(0))
    begin = time.time()
    prob4 = model4.predict(data_val)
    stop = time.time()
    print(stop-begin)
    output = np.zeros((len(prob4)*prob4[1].size,))
    for i in range(len(prob4[1])):
        for j in range(len(prob4)):
            for m in range(len(prob4[1][1])):
                output[((i*len(prob4)+j)*len(prob4[1][1])+m)] = prob4[j][i][m]
    while True:
        print("success")
        label = np.zeros([len(output)])
        label[:] = wave_data[0][0:len(output)]
        time = np.arange(0, len(output)) * (1.0 / 44100)
        pl.subplot(211)
        pl.plot(time, (output-0.5)*65536.0)
        pl.subplot(212)
        pl.plot(time, (label-0.5)*65536.0, c="g")
        pl.xlabel("time (seconds)")
        pl.show()
        samples = []
        for i in range(len(output)):
            sample = (output[i]-0.5)*65536.0
            packed_sample = struct.pack('h', sample)
            samples.append(packed_sample)
        print("predict")
        sample_str = ''.join(samples)
        stream.write(sample_str)
    stream.stop_stream()
    stream.close()
    p.terminate()




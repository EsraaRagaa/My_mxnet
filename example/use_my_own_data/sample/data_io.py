# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
def default_build_vocab(path1, path2):
    dataB = np.load(path1)
    dataX = np.load(path2)
    return (dataB/2+0.5), (dataX/2+0.5)


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self,dataB, dataX, batch_size, data_name='data', label_name='label'):
        super(BucketSentenceIter, self).__init__()
        self.vocab_size = len(dataB)
        self.data_name = data_name
        self.label_name = label_name
        self.batch_size = batch_size
        self.data = dataB
        self.lable = dataX
        self.provide_data = [('data', (self.batch_size,len(self.data[0]),))]
        self.provide_label = [('label', (self.batch_size,len(self.lable[0]),))]
        self.bucket_n_batches = int(len(self.data))
        data = np.zeros((self.batch_size,len(self.data[0]),))
        label = np.zeros((self.batch_size,len(self.lable[0]),))
        self.data_buffer = data
        self.label_buffer =label

    def __iter__(self):
        for idx in range(self.bucket_n_batches):
            data = self.data_buffer
            label = self.label_buffer
            data[:] = self.data[idx]
            label[:] = self.lable[idx]
            data_all = [mx.nd.array(data[:])]
            label_all = [mx.nd.array(label[:])]
            data_names = ['data']
            label_names = ['label']
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch


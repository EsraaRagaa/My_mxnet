# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import cv2
import numpy as np

def default_build_vocab( ):
    print("Start collecting data, please speak something!")
    cap = cv2.VideoCapture(0)
    img_data=np.zeros([600,3,160,120])
    if (cap.isOpened()):
        for i in range(600):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_CUBIC)
                cv2.imshow('frame',frame)
                img_data[i] = np.swapaxes(frame, 0, 2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Stop collecting data!")
    return img_data/255


class SimpleBatch(object):
    def __init__(self, data_names, data , bucket_key):
        self.data = data

        self.data_names = data_names

        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self,img_data, buckets, batch_size,
                 init_states, data_name='data', label_name='label'):
        super(BucketSentenceIter, self).__init__()
        self.vocab_size = len(img_data)
        self.data_name = data_name
        self.label_name = label_name
        self.buckets = buckets
        self.default_bucket_key = buckets
        self.data = img_data
        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('%s/%d' % (self.data_name, t), (self.batch_size, len(self.data[0]), len(self.data[0][0]),
                                                              len(self.data[0][0][0]),))
                             for t in range(self.default_bucket_key)] + init_states
    def make_data_iter_plan(self):
        "make a random data iteration plan"
        self.bucket_n_batches = int((len(self.data)-10) / self.batch_size / self.buckets)
        self.data = self.data[:int(self.bucket_n_batches*self.batch_size*self.buckets)]
        data = np.zeros((self.batch_size, self.buckets, len(self.data[0]), len(self.data[0][0]), len(self.data[0][0][0])))
        self.data_buffer = data

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_batch_bucket in range(self.bucket_n_batches):
            for i_batch in range(self.batch_size):
                for i_bucket in range(self.buckets):
                    data = self.data_buffer
                    idx = (i_batch_bucket * self.batch_size + i_batch)*self.buckets + i_bucket
                    data[i_batch, i_bucket] = self.data[idx]
            data_all = [mx.nd.array(data[:, t])
                        for t in range(self.buckets)] + self.init_state_arrays
            data_names = ['%s/%d' % (self.data_name, t)
                          for t in range(self.buckets)] + init_state_names
            data_batch = SimpleBatch(data_names, data_all, self.buckets)
            yield data_batch


# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import cv2
from pydub import AudioSegment
import numpy as np
def default_build_vocab(path1, path2):
    capture=cv2.VideoCapture(path1)
    ret = True
    img_data=np.zeros([int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))+1,3,160,120])
    i = 0
    while ret:
        ret, imgg = capture.read()
        if ret == False:
            imgg = np.zeros([int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)), 3])
        img = cv2.resize(imgg, (160, 120), interpolation=cv2.INTER_CUBIC)
        img_data[i] = np.swapaxes(img, 0, 2)
        i = i+1
    capture.release()
    cv2.destroyAllWindows()

    mp3_audio = AudioSegment.from_file(path2, format="mp3")
    wave_data = np.fromstring(mp3_audio.raw_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    return img_data/255, (wave_data/65536.0+0.5)


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self,img_data, wave_data, buckets, batch_size,
                 init_states, label_size, data_name='data', label_name='label'):
        super(BucketSentenceIter, self).__init__()
        self.vocab_size = len(img_data)
        self.label_size = label_size
        self.data_name = data_name
        self.label_name = label_name

        self.buckets = buckets

        self.default_bucket_key = buckets

        self.data = img_data
        self.lable = wave_data

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('%s/%d' % (self.data_name, t), (self.batch_size, len(self.data[0]), len(self.data[0][0]),
                                                              len(self.data[0][0][0]),))
                             for t in range(self.default_bucket_key)] + init_states
        self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size,self.label_size))
                              for t in range(self.default_bucket_key)]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        self.bucket_n_batches = int((len(self.data) - 10 - self.batch_size * self.buckets)/5)
        data = np.zeros((self.batch_size, self.buckets, len(self.data[0]), len(self.data[0][0]), len(self.data[0][0][0])))
        label = np.zeros((self.batch_size, self.buckets, self.label_size))
        self.data_buffer = data
        self.label_buffer =label

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_batch_bucket in range(self.bucket_n_batches):
            for i_batch in range(self.batch_size):
                for i_bucket in range(self.buckets):
                    data = self.data_buffer
                    label = self.label_buffer
                    idx = i_batch_bucket * 5 + i_batch*self.buckets + i_bucket
                    data[i_batch, i_bucket] = self.data[idx]
                    label[i_batch, i_bucket, :] = self.lable[0, idx*self.label_size:(idx+1)*self.label_size]

            data_all = [mx.nd.array(data[:, t])
                        for t in range(self.buckets)] + self.init_state_arrays
            label_all = [mx.nd.array(label[:, t])
                         for t in range(self.buckets)]
            data_names = ['%s/%d' % (self.data_name, t)
                          for t in range(self.buckets)] + init_state_names
            label_names = ['%s/%d' % (self.label_name, t)
                           for t in range(self.buckets)]

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets)
            yield data_batch


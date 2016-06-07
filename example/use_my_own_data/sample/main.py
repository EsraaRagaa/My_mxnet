# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
from network import network
from data_io import BucketSentenceIter, default_build_vocab

def Perplexity(label, pred):
    loss = 0.
    global num
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            loss += (pred[i][j] - label[i][j])*(pred[i][j] - label[i][j])
    return loss/pred.shape[0]

if __name__ == '__main__':
    batch_size = 2
    num_epoch = 500
    learning_rate = 0.1

    print(batch_size, num_epoch, learning_rate)
    contexts = [mx.context.gpu(0)]
    dataB,dataX = default_build_vocab("./data/trainingB.npy", "./data/trainingX.npy")
    dataB1,dataX1 = default_build_vocab("./data/validB.npy", "./data/validX.npy")

    data_train = BucketSentenceIter(dataB,dataX,batch_size)
    data_val = BucketSentenceIter(dataB1,dataX1,batch_size)

    symbol = network()
    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=0.0,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    model.fit(X=data_train, eval_data=None,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 1000),
              epoch_end_callback = mx.callback.do_checkpoint('model/BN'))




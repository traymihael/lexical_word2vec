#!/usr/bin/env python
"""Sample script of word embedding model.
This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections

import numpy as np
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions
import word2vec_mi as w2v_mi

cp = cuda.cupy

# CBOWです。こっちはいじった。
class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units, loss_func, ori_con_data):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            # 派生単語と元の単語の初期ベクトルを統一する
            for i in range(len(ori_con_data)):
                self.embed.W.data[ori_con_data[i][0]] = self.embed.W.data[ori_con_data[i][1]]

            self.loss_func = loss_func

    def __call__(self, x, contexts):

        e = self.embed(contexts)
        h = F.sum(e, axis=1) * (1. / contexts.shape[1])
        x = x.astype(cp.int32)
        # ベクトル表示
        for i in range(7,len(self.embed.W.data)-1):
            print(self.embed.W.data[i][0], end=' ')
        print()
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)

        return loss

# skipgramです。こっちはいじってない。というかCBOWと変わらんからいじっても良い。
class SkipGram(chainer.Chain):
    """Definition of Skip-gram Model"""

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.embed.W.data[14] = self.embed.W.data[7]
            self.embed.W.data[15] = self.embed.W.data[9]
            self.loss_func = loss_func

    def __call__(self, x, contexts):
        print('skipgram')
        x = x.astype(cp.int32)
        e = self.embed(contexts)
        batch_size, n_context, n_units = e.shape
        x = F.broadcast_to(x[:, None], (batch_size, n_context))
        e = F.reshape(e, (batch_size * n_context, n_units))
        x = F.reshape(x, (batch_size * n_context,))

        for i in range(7,len(self.embed.W.data)):
            print(self.embed.W.data[i][0], end = ' ')
        print()

        loss = self.loss_func(e, x)
        reporter.report({'loss': loss}, self)
        return loss


class SoftmaxCrossEntropyLoss(chainer.Chain):
    """Softmax cross entropy loss function preceded by linear transformation.
    """

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        with self.init_scope():
            self.out = L.Linear(n_in, n_out, initialW=0)

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


class WindowIterator(chainer.dataset.Iterator):
    """Dataset iterator to create a batch of sequences at different positions.
    This iterator returns a pair of the current words and the context words.
    """

    def __init__(self, dataset, window, batch_size, original_index, repeat=True):
        self.dataset = cp.array(dataset, cp.int32)
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat

        # # window幅を抜いたインデックスと派生単語の入るインデックスを取得
        # self.order = [i+window for i in range(len(dataset)-len(original_index)-window*2 )]
        # self.order = np.array(self.order).astype(np.int32)

        # order is the array which is shuffled ``[window, window + 1, ...,
        self.order = cp.random.permutation(len(dataset) - len(original_index) - window * 2).astype(cp.int32)
        self.order += window

        self.current_position = 0
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False

    def __next__(self):
        """This iterator returns a list representing a mini-batch.
        Each item indicates a different position in the original sequence.
        """

        with open('data_mi.txt', 'r', encoding = 'utf-8',errors = 'ignore') as f:
            data_mi = []
            for line in f:
                line = list(map(int, line.split('\t')))
                data_mi.append(line)

        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]
        w = cp.random.randint(self.window - 1) + 1
        offset = cp.concatenate([cp.arange(-w, 0), cp.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]

        contexts = self.dataset.take(pos)
        center = self.dataset.take(position)

        ins_context = cp.array([])
        ins_center = cp.array([])
        center_num = len(center)
        contexts_num = len(contexts[0])
        last_num = self.dataset[-1]

        for k in range(len(data_mi)):
            dataset_kari = self.dataset.copy()
            for kk in range(len(data_mi[k])-1):
                dataset_kari[data_mi[k][kk+1]] = data_mi[k][0]
            contexts_kari = dataset_kari.take(pos)

            for i in range(center_num):
                if data_mi[k][0] in contexts_kari[i]:
                    kari = contexts_kari[i].copy()
                    for j in range(contexts_num):
                        if kari[j] != data_mi[k][0]:
                            kari[j] = last_num
                    ins_context = cp.append(ins_context, [i+1, kari])
                    ins_center = cp.append(ins_center, [i+1, center[i]])

        ins_context = cp.sort(ins_context, key=lambda x: x[0])
        ins_center.sort(key=lambda x: x[0])

        # contexts = list(contexts)
        # center = list(center)
        #
        # count = 0
        # for i in range(len(ins_context)):
        #     contexts.insert(ins_context[i][0] + count, ins_context[i][1])
        #     center.insert(ins_center[i][0] + count, ins_center[i][1])
        #     count += 1


        count = 0
        for i in range(len(ins_context)):
            cp.insert(contexts, ins_context[i][0] + count, ins_context[i][1])
            cp.insert(center, ins_center[i][0] + count, ins_center[i][1])
            count += 1

        contexts = cp.array((contexts))
        center = cp.array(center)

        if i_end >= len(self.order):
            cp.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, contexts

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


def convert(batch, device):
    center, contexts = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=300, type=int,
                        help='number of units')
    parser.add_argument('--window', '-w', default=3, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=10000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=30, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                        default='cbow',
                        help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int,
                        help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                        default='hsm',
                        help='output model type ("hsm": hierarchical softmax, '
                        '"ns": negative sampling, "original": '
                        'no approximation)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('')

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    # Load the dataset
    train, val, _ = chainer.datasets.get_ptb_words()
    vocab = chainer.datasets.get_ptb_words_vocabulary()
    train, val, _, vocab, original_index, ori_con_data = w2v_mi.get_pair(train, val, _, vocab)
    counts = collections.Counter(train)
    counts.update(collections.Counter(val))
    n_vocab = max(train) + 1

    if args.test:
        train = train[:100]
        val = val[:100]


    index2word = {wid: word for word, wid in six.iteritems(vocab)}

    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % len(train))

    if args.out_type == 'hsm':
        HSM = L.BinaryHierarchicalSoftmax
        tree = HSM.create_huffman_tree(counts)
        loss_func = HSM(args.unit, tree)
        loss_func.W.data[...] = 0
    elif args.out_type == 'ns':
        cs = [counts[w] for w in range(len(counts))]
        loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
        loss_func.W.data[...] = 0
    elif args.out_type == 'original':
        loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
    else:
        raise Exception('Unknown output type: {}'.format(args.out_type))

    # Choose the model
    if args.model == 'skipgram':
        model = SkipGram(n_vocab, args.unit, loss_func)
    elif args.model == 'cbow':
        model = ContinuousBoW(n_vocab, args.unit, loss_func, ori_con_data)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        model.to_gpu()

    # Set up an optimizer
    optimizer = O.Adam()
    optimizer.setup(model)

    # Set up an iterator
    train_iter = WindowIterator(train, args.window, args.batchsize, original_index)
    val_iter = WindowIterator(val, args.window, args.batchsize, original_index, repeat=False)

    # Set up an updater
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)

    # Set up a trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(
        val_iter, model, converter=convert, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    # Save the word2vec model
    with open('word2vec.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word)-1, args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            if i == len(index2word)-1:
                print(i)
                continue
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))


if __name__ == '__main__':
    main()
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

# CBOWです。こっちはいじった。
class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units, loss_func, ori_con_data):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            print('make first embed')
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            print('finish make first embed')
            # 派生単語と元の単語の初期ベクトルを統一する
            for i in range(len(ori_con_data)):
                self.embed.W.data[ori_con_data[i][0]] = self.embed.W.data[ori_con_data[i][1]]
            self.loss_func = loss_func

    def __call__(self, x, contexts):

        e = self.embed(contexts)
        # print(e)
        h = F.sum(e, axis=1) * (1. / contexts.shape[1])
        x = x.astype(np.int32)
        # # ベクトル表示
        # for i in range(7,len(self.embed.W.data)-1):
        #     if i in [7,14]:
        #         print(self.embed.W.data[i][0], end=' ')
        # print()
        # print(h)
        # print(x)
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)

        return loss

# skipgramです。こっちはいじってない。というかCBOWと変わらんからいじっても良い。
class SkipGram(chainer.Chain):
    """Definition of Skip-gram Model"""

    def __init__(self, n_vocab, n_units, loss_func, ori_con_data):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            # 派生単語と元の単語の初期ベクトルを統一する
            for i in range(len(ori_con_data)):
                self.embed.W.data[ori_con_data[i][0]] = self.embed.W.data[ori_con_data[i][1]]

            self.loss_func = loss_func

    def __call__(self, x, contexts):
        # print('skipgram')
        x = x.astype(np.int32)
        e = self.embed(contexts)
        batch_size, n_context, n_units = e.shape
        x = F.broadcast_to(x[:, None], (batch_size, n_context))
        e = F.reshape(e, (batch_size * n_context, n_units))
        x = F.reshape(x, (batch_size * n_context,))

        # for i in range(7,len(self.embed.W.data)):
        #     if i in [121,207,602,603]:
        #         print(self.embed.W.data[i][0], end = ' ')
        # print()

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
        self.dataset = np.array(dataset, np.int32)
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat

        # # window幅を抜いたインデックスと派生単語の入るインデックスを取得
        # self.order = [i+window for i in range(len(dataset)-len(original_index)-window*2 )]
        # self.order = np.array(self.order).astype(np.int32)

        # order is the array which is shuffled ``[window, window + 1, ...,
        self.order = np.random.permutation(len(dataset) - len(original_index) - window * 2).astype(np.int32)
        self.order += window

        self.current_position = 0
        self.flg = 0
        self.flg_fin = 0
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



        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])

        if self.flg == 0:
            position = self.order[i:i_end]
            pos = position[:, None] + offset[None, :]

            contexts = self.dataset.take(pos)
            center = self.dataset.take(position)

            # print(len(center), '-----')


        else:
            last_num = self.dataset[-1]
            # w = np.random.randint(self.window - 1) + 1
            # offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
            contexts = []
            position = []
            # print('start hasei')
            with open('data_mi.txt', 'r', encoding = 'utf-8',errors = 'ignore') as f:
                data_mi = []
                count = 0
                flg_data_num = 1
                # count_all = 0
                # count_fin = 2
                for line in f:
                    if count < int((self.flg-1) * self.batch_size/100):
                        count += len(line.split('\t')) - 1
                        continue
                    line = list(map(int, line.split('\t')))
                    count += len(line) - 1
                    data_mi.append(line)

                    # count_all += 1
                    # if count_all == count_fin:
                    #     flg_data_num = 1
                    #     break

                    if count >= int(self.flg * self.batch_size/100):
                        # print('out')
                        flg_data_num = 0
                        break
                if flg_data_num:
                    self.flg_fin = 1


            for i in range(len(data_mi)):
                hasei_num = data_mi[i][0]
                for j in range(1, len(data_mi[i])):
                    replace_num = data_mi[i][j]

                    for k in range(len(offset)):
                        position.append(replace_num+offset[k])
                        contexts_kari = [last_num for kk in range(len(offset))]
                        contexts_kari[-1-k] = hasei_num
                        contexts.append(contexts_kari)





            # ins_context = []
            # ins_center = []
            # center_num = len(center)
            # contexts_num = len(contexts[0])
            # last_num = self.dataset[-1]
            #
            # for k in range(len(data_mi)):
            #     dataset_kari = self.dataset.copy()
            #     for kk in range(len(data_mi[k])-1):
            #         dataset_kari[data_mi[k][kk+1]] = data_mi[k][0]
            #     contexts_kari = dataset_kari.take(pos)
            #
            #     for i in range(center_num):
            #         if data_mi[k][0] in contexts_kari[i]:
            #             kari = contexts_kari[i].copy()
            #             for j in range(contexts_num):
            #                 if kari[j] != data_mi[k][0]:
            #                     kari[j] = last_num
            #             ins_context.append(kari)
            #             ins_center.append(center[i])
            #
            # contexts = list(contexts)
            # center = list(center)

            # contexts.extend(ins_context)
            # center.extend(ins_center)
            # print('aaa')

            # # 7 8 9 10 14
            # center = [9, 9 ,10, 10]
            # contexts = [[7,10], [14,16],[7,9],[14,16]]
            #
            # center = [9, 10,8,8,8,8,8,8,8,8, 9, 10]
            # contexts = [[7, 10], [7, 9],[10,11],[10,11],[10,8],[8,11],[10,11],[10,11],[10,11],[10,11],[14, 16], [14, 16]]




            position = np.array(position)

            if contexts == []:
                center = np.array([0])
                contexts = np.array([[0 for i in range(2*w)]])
                # center = self.dataset.take(position)
                # contexts = np.array(contexts)
            else:
                center = self.dataset.take(position)
                contexts = np.array(contexts)

        # print(center)
        # print(contexts)
        # print(len(center), '----------------')
        # print(contexts)



        if i_end >= len(self.order):
            if self.flg_fin:
                # print('New')
                np.random.shuffle(self.order)
                self.epoch += 1
                self.is_new_epoch = True
                self.current_position = 0
                self.flg_fin = 0
                self.flg = 0
            else:
                self.is_new_epoch = False
                self.flg += 1
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
    parser.add_argument('--window', '-w', default=5, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
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
    print('load_start')
    train, val, _ = chainer.datasets.get_ptb_words()
    print('loar_finish')
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
        model = SkipGram(n_vocab, args.unit, loss_func, ori_con_data)
    elif args.model == 'cbow':
        print('chose_model cbow')
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
    print('run_start')
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

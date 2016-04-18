from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    args = parser.parse_args()
    train(args)

def train(args):
    print(args)
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            #print("model learning rate is {}".format(model.lr.eval()))
            data_loader.reset_batch_pointer('train')

            state = model.initial_state.eval()
            for b in xrange(data_loader.ntrain):
                start = time.time()
                x, y = data_loader.next_batch('train')

                # tmp = ''
                # for c in x:
                #   for i in c:
                #     tmp += np.array(data_loader.chars)[i]
                # print(tmp)

                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.ntrain + b,
                            args.num_epochs * data_loader.ntrain,
                            e, train_loss, end - start))
                if (e * data_loader.ntrain + b) % args.save_every == 0:
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.ntrain + b)
                    print("model saved to {}".format(checkpoint_path))


            # eval validation loss
            data_loader.reset_batch_pointer('validation')
            validation_state = model.initial_state.eval()
            val_losses = 0
            for n in xrange(data_loader.nvalidation):
                x, y = data_loader.next_batch('validation')
                feed = {model.input_data: x, model.targets: y, model.initial_state: validation_state}
                validation_loss, validation_state = sess.run([model.cost, model.final_state], feed)
                val_losses += validation_loss

            validation_loss = val_losses / data_loader.nvalidation
            print("validation loss is {}".format(validation_loss))

if __name__ == '__main__':
    main()

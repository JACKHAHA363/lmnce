from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

import utils
import sys
import pickle

from word2vec import Word2Vec 

MAX_LENGTH = 3000000

class Word2VecNCE(Word2Vec):
    """ This use nce to train data. This is our baseline """

    def __init__(self, *args, **kwargs):
        super(Word2VecNCE, self).__init__(*args, **kwargs)
        
        self.noise_unigram = 1.0 / self.vocab_size * tf.ones([self.vocab_size])
        self.prob_buf = np.random.randint(
                low=1,high=self.vocab_size,
                size=MAX_LENGTH)
 
    def nceloghtrue(self, logits):
        z = tf.nn.embedding_lookup(self.z, self.currLabels)
        pm = tf.exp(logits) / z
        pn = tf.gather(self.noise_unigram, self.currLabels)
        log_numerator = 0
        log_denominator = tf.log(1 + self.num_sample * pn / pm)
        return log_numerator-log_denominator
    
    def nceloghfalse(self, logits):
        z = tf.nn.embedding_lookup(self.z, self.negLabels)
        pm = tf.exp(logits) / z
        pn = tf.gather(self.noise_unigram, self.negLabels)
        log_numerator = 0
        log_denominator = tf.log(1 + pm / (self.num_sample * pn))
        return log_numerator-log_denominator

    def likelihood(self):
        true_logits = self.getlogits(
                self.currWords, self.currLabels,
                self.sm_w_t, self.sm_b, self.emb)
        false_logits = self.getlogits(
                self.negWords, self.negLabels,
                self.sm_w_t, self.sm_b, self.emb)
        ll_true = tf.reduce_sum(self.nceloghtrue(true_logits), 0)/self.batch_size
        ll_false = tf.reduce_sum(self.nceloghfalse(false_logits), 0)/(self.batch_size)
        return ll_true + ll_false
        
    def generate_next_feed_dict(self, data):
        batch_words, batch_labels = self.generate_next_batch(data)

        index = np.random.randint(MAX_LENGTH, size=self.batch_size * self.num_sample)
        self.sample = self.prob_buf[index]      
        
        # generate NCE Samples
        sampled_currWords = np.repeat(batch_words, self.num_sample)
        feeddict = {
                self.currWords : batch_words, 
                self.currLabels : batch_labels,
                self.negWords : sampled_currWords,
                self.negLabels : self.sample
                }
        return feeddict
    
    def init_noise(self, noisefile):
        if noisefile is not None:
            with open(noisefile, 'rb') as f:
                unigram = pickle.load(f)
                self.noise_unigram = tf.constant(unigram, name="noise_unigram")
            
            # sample from given unigram
            pnt = 0
            for i in range(self.vocab_size):
                current_length = int(MAX_LENGTH * unigram[i])
                self.prob_buf[pnt:pnt + current_length] = i
                pnt += current_length

def main(args):
    if len(args) < 2:
        return
    savefile = args[0]
    lr_init = float(args[1])
    
    with open("./data/text8batch.dat", "rb") as f:
        train_data = pickle.load(f)
    
    with open("./data/text8valid.dat", "rb") as f:
        test_data = pickle.load(f)

    with open("./data/text8_small.dat", "rb") as f:
        data = pickle.load(f)
    vocab_size = data["vocab_size"]   

    config = tf.ConfigProto(
            intra_op_parallelism_threads = 8,
            inter_op_parallelism_threads = 8)

    with tf.Session(config=config) as sess:
        model = Word2VecNCE(
                emb_dim=10, vocab_size=vocab_size, num_sample=10,
                word_per_batch=1024, window=1, sess=sess)
        model.train(train_data=train_data, test_data=test_data, num_epoch=30, savefile=savefile, lr_init=lr_init)

if __name__ == "__main__":
    main(sys.argv[1:])

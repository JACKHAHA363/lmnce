from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import numexpr as ne
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

import utils
import sys
import pickle

ZERO = 1e-5

class Word2Vec():
    """ a parent class for word2vec """

    def __init__(self, emb_dim, vocab_size, num_sample, word_per_batch, window, sess):
        self.sess = sess
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_sample = num_sample
        self.word_per_batch = word_per_batch
        self.batch_size = 2 * window * word_per_batch
        self.window = window
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)
        self.lr_init = 0.1
        self.epoch = 0
        self.step = 0
        self.data_index = 0
        self.batch_index = 0

        init_width = 0.5/emb_dim
        # Embedding: [vocab_size, emb_dim]
        self.emb = tf.Variable(
                tf.random_uniform(
                    [self.vocab_size, self.emb_dim], -init_width, init_width),
                name="emb")
       
        # Softmax weight aka the context weight
        self.sm_w_t = tf.Variable(
                tf.ones([self.vocab_size, self.emb_dim]) * ZERO,
                name="sm_w_t")

        # Softmax bias
        self.sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

        # PlaceHolders for the data and negative sample
        self.currWords = tf.placeholder(
                dtype=tf.int32, 
                shape=[self.batch_size], 
                name="currWords")
        
        self.currLabels = tf.placeholder(
                dtype=tf.int32, 
                shape=[self.batch_size], 
                name="currLabels")
        
        self.negWords = tf.placeholder(
                dtype=tf.int32,
                shape=[self.batch_size * self.num_sample],
                name="negWords")
        
        self.negLabels = tf.placeholder(
                dtype=tf.int32,
                shape=[self.batch_size * self.num_sample],
                name="negLabels")
        
        # normalized constant
        self.z = tf.Variable(tf.ones([vocab_size]) * self.vocab_size, name="z", trainable=True)
    
        # This is used to hold the value of samples. 
        # Could be used as negLabels or as weights in importance sampling
        self.sample = np.random.randint(
                low=1,high=self.vocab_size,
                size=self.batch_size * self.num_sample)
        
    def getlogits(self, words, labels, weight, bias, emb):
        """ Get the logits for the given data and embedding """
    
        w = tf.nn.embedding_lookup(weight, words)
        b = tf.nn.embedding_lookup(bias, labels)
        word_emb = tf.nn.embedding_lookup(emb, labels)
        result = tf.reduce_sum(tf.mul(word_emb, w), 1) + b
        #result = tf.maximum(tf.minimum(result, 10), -10)
        return result
    
    def likelihood(self):
        """ This function get the likelihood of the model """
    
        raise NotImplementedError()
    
    def generate_next_batch(self, data):
        """ This function generate the next batch """ 
        
        batch_words = np.array(data[self.batch_lookup[self.batch_index]][0])
        batch_labels = np.array(data[self.batch_lookup[self.batch_index]][1])
        self.batch_index += 1
        if self.batch_index == len(data) - 1:
            self.epoch += 1
        return batch_words, batch_labels
    
    def generate_next_feed_dict(self, data):
        """ This function generate the feed_dict for current iteration """
        
        raise NotImplementedError()
    
    def optimize(self, loss):
        """ It use gradient descend with linearly decaying learning rate"""
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        return optimizer.minimize(loss)
    
    def numerical_error_handing(self, feeddict):
        """ This is function is used to examine the state when there is a numerical error """

        raise NotImplementedError()

    def init_noise(self, noisefile):
        raise NotImplementedError()

    def train(self, train_data, test_data, num_epoch, savefile, lr_init, noisefile=None):
        """ This train the data and save the result """
        
        self.test_data = test_data
        self.lr_init = lr_init
        self.num_batches = len(train_data)
        loss = -self.likelihood()
        logpp = tf.placeholder(tf.float32)
        logPP_summary = tf.scalar_summary("log perplexity", logpp)
        loss_summary = tf.scalar_summary("Loss", loss)
        #z_summmary = tf.histogram_summary("z", self.z)
        writer = tf.train.SummaryWriter(savefile+str(lr_init), self.sess.graph)
        
        self.train_op = self.optimize(loss)
        running = True
        
        if noisefile is not None:
            self.init_noise(noisefile)


        self.sess.run(tf.initialize_all_variables())
    
        prev_pp = self.evaluate()
        after_pp = 0

        self.step = 0
        lr = self.lr_init
        try:
            for _ in xrange(num_epoch):
                logPP_sum = self.sess.run(logPP_summary, feed_dict={logpp : prev_pp})
                writer.add_summary(logPP_sum, self.epoch)

                if running == False:
                    break
                
                if lr < 1e-5:
                    break                

                print("start epoch " + str(self.epoch))
                self.batch_lookup = np.random.permutation(len(train_data)-1)
                self.batch_index = 0
                init_epoch = self.epoch
                while init_epoch == self.epoch:
                    feeddict = self.generate_next_feed_dict(train_data)
                    feeddict[self.lr] = lr
                    self.sess.run(self.train_op, feed_dict=feeddict)
            
                    if self.step % 100 == 0:
                        loss_sum, theloss= self.sess.run(
                                [loss_summary, loss], 
                                feed_dict=feeddict)
                        print("at step " + str(self.step) + " the loss is "+ str(theloss))
                        print("learng rate is " + str(lr))
                            
                        writer.add_summary(loss_sum, self.step)                        
           
                        if np.isnan(theloss) or theloss < -40:
                            self.numerical_error_handing(feeddict)
                            running = False
                            break

                    self.step += 1
                
                after_pp = self.evaluate()
                print("pp: " + str(after_pp))
                if after_pp > prev_pp:
                    lr /=  2.0
                prev_pp = after_pp
               #lr = self.lr_init * np.maximum(0.0001, 1.0 - float(self.epoch * self.num_batches+self.batch_index)/(num_epoch * self.num_batches))
            logPP_sum = self.sess.run(logPP_summary, feed_dict={logpp : prev_pp})
            writer.add_summary(logPP_sum, self.epoch)

        except KeyboardInterrupt:
            self.numerical_error_handing(self.generate_next_feed_dict(train_data))

    def evaluate(self):
        """ This function would build the graph to evalute the log perplexity of the test data """
        weight, bias, emb = self.sess.run([self.sm_w_t, self.sm_b, self.emb])
        return utils.pp(weight, bias, emb, self.test_data)

    def save_model(self, savefile):
        weight, bias, emb, z = self.sess.run([self.sm_w_t, self.sm_b, self.emb, self.z])
        with open(savefile, 'wb') as f:
            pickle.dump({"sm_w_t" : weight, "sm_b" : bias, "emb" : emb, "z" : z}, f)
        print("model saved")
 

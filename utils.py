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
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle

# Download data
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  url = 'http://mattmahoney.net/dc/'
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# Read the data into a string.
def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name).split()
  f.close()

# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

# the main function to generate train and test data
def gen_data_small(vocab_size):
    filename = maybe_download('text8.zip', 31344016)
    words = read_data(filename)
    print('Data size', len(words))
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size)
    del words 
    data = np.array(data)
    data = data[data.nonzero()]
    data = data[0 : int(0.1 * np.size(data))]
    train_end = int(0.95 * np.size(data))
    train_data = data[0 : train_end]
    test_data = train_data[0 : int(np.size(data))/5]
    
    count = np.zeros([vocab_size])
    for word in train_data:
        count[word] += 1
    unigram = count / np.sum(count)
    return train_data, test_data, unigram

def gen_analogy_data():
    with open("./analogy/word_relationship.questions", "r") as f:
        q_content = f.read().splitlines()
    with open("./analogy/answer.txt", "r") as f:
        a_content = f.read().splitlines()
    questions = []
    for i in range(np.size(a_content)):
        words = [ x.encode('ascii') for x in q_content[i].split() ]
        words.append(a_content[i].encode('ascii'))
        questions.append(words)
    return questions

def make_batched_dataset(window, data, batch_size=None):
    data_index = window
    if batch_size is None:
        words = []
        labels = []
        while True:
            curr = data[data_index]
            lower = data_index - window
            upper = data_index + window
            if upper == len(data):
                break
            else:
                context_pos = [x for x in range(lower, data_index)] + [x for x in range(data_index+1, upper+1)]
                for j in context_pos:
                    words.append(curr)
                    labels.append(data[j])
                data_index += 1
        return (words, labels)   
    else:
        batches = []
        stop = 0
        while stop == 0:
            # generate one batch
            batch_words = []
            batch_labels = []
            num_words = 0
            while num_words < batch_size:
                curr = data[data_index]
                lower = data_index - window
                upper = data_index + window
                if upper == len(data):
                    stop = 1
                    break
                else:
                    context_pos = [x for x in range(lower, data_index)] + [x for x in range(data_index+1, upper+1)]
                    for j in context_pos:
                        batch_words.append(curr)
                        batch_labels.append(data[j])

                    data_index += 1
                    num_words += 1
            
            batches.append((batch_words, batch_labels))

        return batches

def pp(weight, bias, emb, test_data):

    ''' probs[i][j] = P[ word_j | word_i] '''
    logits = np.dot(weight, emb.transpose()) + bias
    expo = ne.evaluate("exp(logits)")
    z = np.sum(expo, axis=1)
    tmp = expo.transpose()
    probs = ne.evaluate('tmp / z')
    probs = probs.transpose() 
    del tmp
    
    lookup = probs[test_data[0], test_data[1]]
    result = ne.evaluate('-log(lookup)')      
    return np.sum(result) / len(result)


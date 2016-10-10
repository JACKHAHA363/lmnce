from utils import gen_data_small
from utils import make_batched_dataset
import numpy as np
import pickle

train_data, test_data, unigram = gen_data_small(50000)
print(np.size(train_data))
print(np.size(test_data))
pickle.dump({"train" : train_data, "test" : test_data, "vocab_size" : 50000}, open("./data/text8_small.dat", "wb"))

batches = make_batched_dataset(1, train_data, 1024)
pickle.dump(batches, open("./data/text8batch.dat", "wb"), protocol=4)

test = make_batched_dataset(1, train_data)
pickle.dump(test, open("./data/text8valid.dat", "wb"), protocol=4)


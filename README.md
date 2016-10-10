# Language Model with NCE
 
Modified from https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/embedding/word2vec.py

This is a experimental project aiming to see the training result using different noise selection in NCE.

## Get Data Ready
Create a `data` folder first. Then run 
```
python gen_data.py
```
This will preprocess the text8 dataset in suitable format

## Usage
```
python word2vec_nce.py [path_to_result] [learning_rate]
```

or you see see the trained result in `simple_LM` by

```
tensorboard --logdir=simple_LM
```

## TODO
Trying to use pretrained models as noise

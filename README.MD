## English - Vietnamese Machine Translation Using Transformer

### I. Introduction:

Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

<img src = "data/image.png">


### 2. Training:

This implementation provides one way to retrain transfomer on English- Vietnamese Machine Translation Dataset ( IWSLT2015 English-Vietnamese). If you want to retrain it, please use this command.

```console
python train.py
```

These hyper paramters was set by default, if you want to change them, you can use argparse command to change them.

### 3.Dataset:

The Dataset contains 3 folders. Train (133k sentences), Dev( 1200 sentences) and Test( 1200 sentences).

### 4.Some Results:



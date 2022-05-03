# Source: https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer

# Subword tokenizers

import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

"""## Download the dataset

Fetch the Sanskrit/English translation dataset
"""

# (2hi) SA-HI 
# (2en) SA -> EN
# (2hi) EN -> HI

from datasets import load_dataset

dataset_itihasa = load_dataset("rahular/itihasa")
dataset_iitb = load_dataset("cfilt/iitb-english-hindi")
dataset_itihasa['train'][0]

dataset_itihasa.set_format(type='pd')
dataset_iitb.set_format(type='pd')

dataset_itihasa

dataset_iitb

dataset_iitb['train']['translation'][0]

import pandas as pd

df_train1 = pd.DataFrame(dataset_itihasa['train']['translation'].to_list())[['sn','en']]
df_valid1 = pd.DataFrame(dataset_itihasa['validation']['translation'].to_list())[['sn','en']]

# df_train1['sn'] = df_train1['sn'].apply(lambda x: '[2EN] '+x)
# df_valid1['sn'] = df_valid1['sn'].apply(lambda x: '[2EN] '+x)

# df_train2 = pd.DataFrame(dataset_iitb['train']['translation'].to_list())[['en','hi']]
# df_valid2 = pd.DataFrame(dataset_iitb['validation']['translation'].to_list())[['en','hi']]

# df_train2['en'] = df_train2['en'].apply(lambda x: '[2HI] '+x)
# df_valid2['en'] = df_valid2['en'].apply(lambda x: '[2HI] '+x)

df_train1.columns = ['inp_lang', 'tar_lang']
# df_train2.columns = ['inp_lang', 'tar_lang']

df_valid1.columns = ['inp_lang', 'tar_lang']
# df_valid2.columns = ['inp_lang', 'tar_lang']

df_train1.head()

# df_train = pd.concat([df_train1, df_train2])
# df_valid = pd.concat([df_valid1, df_valid2])

df_train = df_train1.copy()
df_valid = df_valid1.copy()

df_train.reset_index(drop=True, inplace=True)
df_valid.reset_index(drop=True, inplace=True)

df_train

train_examples = tf.data.Dataset.from_tensor_slices((df_train['inp_lang'], df_train['tar_lang']))
val_examples = tf.data.Dataset.from_tensor_slices((df_valid['inp_lang'], df_valid['tar_lang']))

# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
#                                as_supervised=True)
# train_examples1, val_examples1 = examples['train'], examples['validation']

# train_examples1.take(1)

"""This dataset produces Sanskrit/English sentence pairs:"""

train_examples.take(1)

for inp_lang, tar_lang in train_examples.take(2):
  print("INP: ", inp_lang.numpy().decode('utf-8'))
  print("TAR:   ", tar_lang.numpy().decode('utf-8'))

"""Note a few things about the example sentences above:
* They're lower case.
* There are spaces around the punctuation.
* It's not clear if or what unicode normalization is being used.
"""

train_tar_lang = train_examples.map(lambda inp_lang, tar_lang: tar_lang)
train_inp_lang = train_examples.map(lambda inp_lang, tar_lang: inp_lang)

"""## Generate the vocabulary

This section generates a wordpiece vocabulary from a dataset. If you already have a vocabulary file and just want to see how to build a `text.BertTokenizer` or `text.Wordpiece` tokenizer with it then you can skip ahead to the [Build the tokenizer](#build_the_tokenizer) section.

Note: The vocabulary generation code used in this tutorial is optimized for **simplicity**. If you need a more scalable solution consider using the Apache Beam implementation available in [tools/wordpiece_vocab/generate_vocab.py](https://github.com/tensorflow/text/blob/master/tensorflow_text/tools/wordpiece_vocab/generate_vocab.py)

The vocabulary generation code is included in the `tensorflow_text` pip package. It is not imported by default , you need to manually import it:
"""

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

"""The `bert_vocab.bert_vocab_from_dataset` function will generate the vocabulary. 

There are many arguments you can set to adjust its behavior. For this tutorial, you'll mostly use the defaults. If you want to learn more about the options, first read about [the algorithm](#algorithm), and then have a look at [the code](https://github.com/tensorflow/text/blob/master/tensorflow_text/tools/wordpiece_vocab/bert_vocab_from_dataset.py).

This takes about 2 minutes.
"""

bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[2SN]", "[2EN]", "[2HI]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 10000, # 20000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

df_train_small = df_train.copy()
train_examples_small = tf.data.Dataset.from_tensor_slices((df_train_small['inp_lang'], df_train_small['tar_lang']))

train_tar_lang_small = train_examples_small.map(lambda inp_lang, tar_lang: tar_lang)
train_inp_lang_small = train_examples_small.map(lambda inp_lang, tar_lang: inp_lang)

for i in train_inp_lang_small.take(1):
  print(i.numpy().decode('utf-8'))

inp_lang_vocab = bert_vocab.bert_vocab_from_dataset(
    train_inp_lang_small.batch(1000).prefetch(2),
    **bert_vocab_args
)

"""Here are some slices of the resulting vocabulary."""

print(inp_lang_vocab[:10])
print(inp_lang_vocab[100:110])
print(inp_lang_vocab[1000:1010])
print(inp_lang_vocab[-10:])

len(inp_lang_vocab)

"""Write a vocabulary file:"""

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

write_vocab_file('inp_lang_vocab.txt', inp_lang_vocab)

"""Use that function to generate a vocabulary from the english data:"""

tar_lang_vocab = bert_vocab.bert_vocab_from_dataset(
    train_tar_lang_small.batch(1000).prefetch(2),
    **bert_vocab_args
)

len(tar_lang_vocab)

print(tar_lang_vocab[:10])
print(tar_lang_vocab[100:110])
print(tar_lang_vocab[1000:1010])
print(tar_lang_vocab[-10:])

"""Here are the two vocabulary files:"""

write_vocab_file('tar_lang_vocab.txt', tar_lang_vocab)

"""## Build the tokenizer
<a id="build_the_tokenizer"></a>

The `text.BertTokenizer` can be initialized by passing the vocabulary file's path as the first argument (see the section on [tf.lookup](#tf.lookup) for other options):
"""

inp_lang_tokenizer = text.BertTokenizer('inp_lang_vocab.txt', **bert_tokenizer_params)
tar_lang_tokenizer = text.BertTokenizer('tar_lang_vocab.txt', **bert_tokenizer_params)

"""Now you can use it to encode some text. Take a batch of 3 examples from the english data:"""

for inp_lang_examples, tar_lang_examples in train_examples.batch(3).take(1):
  for ex in inp_lang_examples:
    print(ex.numpy().decode('utf-8'))

for inp_lang_examples, tar_lang_examples in train_examples.batch(3).take(1):
  for ex in tar_lang_examples:
    print(ex.numpy().decode('utf-8'))

"""Run it through the `BertTokenizer.tokenize` method. Initially, this returns a `tf.RaggedTensor` with axes `(batch, word, word-piece)`:"""

# Tokenize the examples -> (batch, word, word-piece)
token_batch = inp_lang_tokenizer.tokenize(inp_lang_examples)

# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)

for ex in token_batch.to_list():
  print(ex)

"""If you replace the token IDs with their text representations (using `tf.gather`) you can see that in the first example the words `"searchability"` and  `"serendipity"` have been decomposed into `"search ##ability"` and `"s ##ere ##nd ##ip ##ity"`:"""

# Lookup each token id in the vocabulary.
txt_tokens = tf.gather(inp_lang_vocab, token_batch)
# Join with spaces.
tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)

"""To re-assemble words from the extracted tokens, use the `BertTokenizer.detokenize` method:"""

words = inp_lang_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)

"""> Note: `BertTokenizer.tokenize`/`BertTokenizer.detokenize` does not round
trip losslessly. The result of `detokenize` will not, in general, have the
same content or offsets as the input to `tokenize`. This is because of the
"basic tokenization" step, that splits the strings into words before
applying the `WordpieceTokenizer`, includes irreversible
steps like lower-casing and splitting on punctuation. `WordpieceTokenizer`
on the other hand **is** reversible.

## Customization and export

This tutorial builds the text tokenizer and detokenizer used by the [Transformer](https://tensorflow.org/text/tutorials/transformer) tutorial. This section adds methods and processing steps to simplify that tutorial, and exports the tokenizers using `tf.saved_model` so they can be imported by the other tutorials.

### Custom tokenization

The downstream tutorials both expect the tokenized text to include `[START]` and `[END]` tokens.

The `reserved_tokens` reserve space at the beginning of the vocabulary, so `[START]` and `[END]` have the same indexes for both languages:
"""

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged, input):
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  if input: # no need of start token 
    return tf.concat([ragged, ends], axis=1)
  return tf.concat([starts, ragged, ends], axis=1)

words = inp_lang_tokenizer.detokenize(add_start_end(token_batch, input=True))
tf.strings.reduce_join(words, separator=' ', axis=-1)

"""### Custom detokenization

Before exporting the tokenizers there are a couple of things you can cleanup for the downstream tutorials:

1. They want to generate clean text output, so drop reserved tokens like `[START]`, `[END]` and `[PAD]`.
2. They're interested in complete strings, so apply a string join along the `words` axis of the result.  
"""

def cleanup_text(reserved_tokens, token_txt):
  # Drop the reserved tokens, except for "[UNK]".
  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)
    
  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

  # Join them into strings.
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)

  return result

tar_lang_examples.numpy()

token_batch = tar_lang_tokenizer.tokenize(tar_lang_examples).merge_dims(-2,-1)
words = tar_lang_tokenizer.detokenize(token_batch)
words

cleanup_text(reserved_tokens, words).numpy()

"""### Export

The following code block builds a `CustomTokenizer` class to contain the `text.BertTokenizer` instances, the custom logic, and the `@tf.function` wrappers required for export.
"""

def add_start_end(ragged, input):
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  # if input: # no need of start token 
  #   return tf.concat([ragged, ends], axis=1)
  return tf.concat([starts, ragged, ends], axis=1)

class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path, input_lang):
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
    self._reserved_tokens = reserved_tokens
    self._vocab_path = tf.saved_model.Asset(vocab_path)

    vocab = pathlib.Path(vocab_path).read_text().splitlines()
    self.vocab = tf.Variable(vocab)
    self.input_lang = input_lang

    ## Create the signatures for export:   

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))
    
    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()
    
  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = add_start_end(enc, self.input_lang)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)

"""Build a `CustomTokenizer` for each language:"""

tokenizers = tf.Module()
tokenizers.inp_lang = CustomTokenizer(reserved_tokens, 'inp_lang_vocab.txt', input_lang=True)
tokenizers.tar_lang = CustomTokenizer(reserved_tokens, 'tar_lang_vocab.txt', input_lang=False)

"""Export the tokenizers as a `saved_model`:"""

model_name = 'translate_sa_en_converter'
tf.saved_model.save(tokenizers, model_name)
"""### Run inference

The following steps are used for inference:

* Encode the input sentence using the Input Language tokenizer (`tokenizers.inp_lang`). This is the encoder input.
* The decoder input is initialized to the `[START]` token.
* Calculate the padding masks and the look ahead masks.
* The `decoder` then outputs the predictions by looking at the `encoder output` and its own output (self-attention).
* Concatenate the predicted token to the decoder input and pass it to the decoder.
* In this approach, the decoder predicts the next token based on the previous tokens it predicted.

Note: The model is optimized for _efficient training_ and makes a next-token prediction for each token in the output simultaneously. This is redundant during inference, and only the last prediction is used.  This model can be made more efficient for inference if you only calculate the last prediction when running in inference mode (`training=False`).
"""

# Credits: https://www.tensorflow.org/text/tutorials/transformer

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd

import tensorflow_addons as tfa

import nltk
from nltk.translate.chrf_score import sentence_chrf

from datasets import load_dataset

from tqdm import tqdm

from functools import lru_cache
from itertools import product

class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length):
    # input sentence is Input Language, hence adding the start and end token
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.inp_lang.tokenize(sentence).to_tensor()

    # print(sentence)

    encoder_input = sentence

    # As the output language is Target Language, initialize the output with the
    # Target Language start token.

    start_end = self.tokenizers.tar_lang.tokenize([''])[0]
    
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    all_predictions = []

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      # print(predictions)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      all_predictions.append(predictions)

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)

    text = self.tokenizers.tar_lang.detokenize(output)[0]  # shape: ()
    tokens = self.tokenizers.tar_lang.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text, tokens, attention_weights, all_predictions

"""Note: This function uses an unrolled loop, not a dynamic loop. It generates `MAX_TOKENS` on every call. Refer to [NMT with attention](nmt_with_attention.ipynb) for an example implementation with a dynamic loop, which can be much more efficient.

Create an instance of this `Translator` class, and try it out a few times:
"""

def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')


# Source: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

from math import log
from numpy import array
from numpy import argmax
 
# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score - log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences
 
# # define a sequence of 10 words over a vocab of 5 words
# data = [[0.1, 0.2, 0.3, 0.4, 0.5],
# 		[0.5, 0.4, 0.3, 0.2, 0.1],
# 		[0.1, 0.2, 0.3, 0.4, 0.5],
# 		[0.5, 0.4, 0.3, 0.2, 0.1],
# 		[0.1, 0.2, 0.3, 0.4, 0.5],
# 		[0.5, 0.4, 0.3, 0.2, 0.1],
# 		[0.1, 0.2, 0.3, 0.4, 0.5],
# 		[0.5, 0.4, 0.3, 0.2, 0.1],
# 		[0.1, 0.2, 0.3, 0.4, 0.5],
# 		[0.5, 0.4, 0.3, 0.2, 0.1]]
# data = array(data)
# # decode sequence
# result = beam_search_decoder(data, 3)
# # print result
# for seq in result:
# 	print(seq)


# Greedy decoding: 

# all_sentence_chrf, all_ter, all_bleu = [], [], []

# for ind, row in tqdm(df_filter.iterrows()):

#   if REVERSE_TRANSLATION: # en input
#     sentence = row['tar_lang']
#     ground_truth = row['inp_lang']
#   else:
#     sentence = row['inp_lang']
#     ground_truth = row['tar_lang']

#   translated_text, translated_tokens, attention_weights, all_predictions = translator(tf.constant(sentence))

#   ref = ground_truth.split()
#   hyp = str(translated_text.numpy().decode("utf-8")).split()
#   all_sentence_chrf.append(sentence_chrf(ref, hyp, min_len=1, max_len=4))

#   all_ter.append(ter(hyp, ref))

#   all_bleu.append(nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=(0.5, 0.5)))

# print(all_sentence_chrf[0], all_ter[0], all_bleu[0])

# df_metrics = pd.DataFrame(list(zip(all_sentence_chrf, all_ter, all_bleu)),
#                columns =['chrf', 'ter', 'bleu'])

# df_metrics.to_csv("df_test_metrics.csv", index=False)


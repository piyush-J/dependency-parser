"""
Authors:

Rashi Kumar
Piyush Jha
"""

import sys
from collections import deque

import conllu
import nltk.translate.gleu_score as gleu
import numpy as np
import tensorflow as tf
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import sentence_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Lambda


def extract_file(lines_num, length, lines_num1, dup, filename):
    tokens = []
    with open(filename, "r") as xh:
        hin_lines = xh.readlines()
        for i in range(len(hin_lines)):
            tokens.append([])
            for t in indic_tokenize.trivial_tokenize(hin_lines[i]):
                tokens[i].append(t)
                if len(tokens[i]) > 70:
                    lines_num.append(i)
                    length.append(len(tokens[i]))

    for z in lines_num:
        if z not in dup:
            lines_num1.append(z)
            dup.append(z)

    length.sort()

    lin = []
    for t in indic_tokenize.trivial_tokenize(hin_lines[1]):
        lin.append(t)

    return lin, length


def preprocess_lines(xa, xh, lines_num1):
    file_lines = {}
    initial_line = 0

    for line in xa:
        file_lines[initial_line] = line.strip()
        initial_line += 1

    for line_number, line_content in file_lines.items():
        if line_number not in lines_num1:  # duplicate_line list from eng list above
            xh.write(line_content + '\n')


def preorderTraversal(root):
    word_order = {}  # as we don't get the original sentence token order using parsetree
    Stack = deque([])
    # 'Preorder'-> contains all the
    # visited nodes.
    Preorder = []
    Preorder.append(root.token['form'])
    word_order[root.token['form']] = root.token['id']
    Stack.append(root)
    while len(Stack) > 0:
        flag = 0
        if len((Stack[len(Stack) - 1]).children) == 0:
            X = Stack.pop()
        else:
            Par = Stack[len(Stack) - 1]
        for i in range(0, len(Par.children)):
            if Par.children[i].token['form'] not in Preorder:
                flag = 1
                Stack.append(Par.children[i])
                Preorder.append(Par.children[i].token['form'])
                word_order[Par.children[i].token['form']] = Par.children[i].token['id']
                break;
        if flag == 0:
            Stack.pop()
    return Preorder, word_order


def postorderTraversal(root):
    word_order = {}  # as we don't get the original sentence token order using parsetree
    Postorder = []
    if not root:
        return []
    Stack = [root]
    last = None

    while Stack:
        root = Stack[-1]
        if not root.children or last and (last in root.children):
            Postorder.append(root.token['form'])
            word_order[root.token['form']] = root.token['id']
            Stack.pop()
            last = root
        else:
            for children in root.children[::-1]:
                Stack.append(children)
    return Postorder, word_order


def process_indic_lib(filename):
    # The path to the local git repo for Indic NLP library
    INDIC_NLP_LIB_HOME = r"indic_nlp_library"

    # The path to the local git repo for Indic NLP Resources
    INDIC_NLP_RESOURCES = r"indic_nlp_resources"

    # Add library to Python path
    sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))

    # Set environment variable for resources folder
    common.set_resources_path(INDIC_NLP_RESOURCES)

    # for target test file (hindi test file)

    hinditext_file = open(filename, "r", encoding="utf-8")
    hindilines = hinditext_file.readlines()
    print(len(hindilines))
    tokens = []
    tokens1 = []
    hin_sent_tok = []
    for i in range(len(hindilines)):
        hindilines1 = "sos " + hindilines[i].strip("\n") + " eos"
        tokens.append([])
        for t in indic_tokenize.trivial_tokenize(hindilines1):
            tokens[i].append(t)

        hin_sent_tok.append(sentence_tokenize.sentence_split(hindilines1, lang='hi'))
    return hin_sent_tok


def preprocess_using_parse_tree(filename):
    data_file = open(filename, "r", encoding="utf-8")
    all_corpus_text_test_s = []
    all_corpus_index_test_order_s = []
    reduce_count = 148

    for tokentree in conllu.parse_tree_incr(data_file):
        print(tokentree)
        print(tokentree.print_tree())
        print(tokentree.metadata['text'])
        try:
            pre_order, normal_order = preorderTraversal(tokentree)
        except Exception:
            continue
        post_order, _ = postorderTraversal(tokentree)
        normal_order = sorted(normal_order, key=normal_order.get)  # sort values and get the key, i.e., words
        normal_order = [w.lower() for w in normal_order]
        pre_order = [w.lower() for w in pre_order]
        post_order = [w.lower() for w in post_order]

        print("Normal: ", normal_order, "\nPre-order: ", pre_order, "\nPost-order: ", post_order)

        pre_order_index_input = np.array([pre_order.index(item) for item in normal_order]).reshape(-1, )
        post_order_index_input = np.array([post_order.index(item) for item in normal_order]).reshape(-1, )
        print("\nPre-orderindex: ", pre_order_index_input, "\nPost-orderindex: ", post_order_index_input)

        all_corpus_text_test_s.append([normal_order, pre_order, post_order])
        all_corpus_index_test_order_s.append([pre_order_index_input, post_order_index_input])

        reduce_count -= 1
        if reduce_count <= 0:
            break

        return all_corpus_text_test_s, all_corpus_index_test_order_s


def padding_helper(all_corpus_index_order_1D, max_length_sr):
    all_corpus_index_order_1D_pad = []
    for sent in all_corpus_index_order_1D:
        sent_len = len(sent)
        print(sent_len)
        if sent_len < max_length_sr:  # logic for padding the index
            new_sent = list(sent) + list(range(sent_len, max_length_sr))
        all_corpus_index_order_1D_pad.append(np.array(new_sent))
    all_corpus_index_order_1D_pad = np.array(all_corpus_index_order_1D_pad)
    all_corpus_index_order_1D_pad_reshape = all_corpus_index_order_1D_pad.reshape(-1, 2,
                                                                                  all_corpus_index_order_1D_pad.shape[
                                                                                      -1])
    return all_corpus_index_order_1D_pad_reshape


def create_input(all_corpus_index_order_1D_pad_reshape, constant_batch_size, corpus_pad_sr):
    input_set_X1, input_set_X2, input_set_X3, input_set_X4, input_set_X5 = [], [], [], [], []
    for sample_ind in range(5120, 6400):
        # for sample_ind in range(len(X_train)):#for different test data set
        preorder_sampled = np.insert(
            all_corpus_index_order_1D_pad_reshape[sample_ind][0].reshape(corpus_pad_sr.shape[-1], 1), 0,
            sample_ind % constant_batch_size, axis=1)
        postorder_sampled = np.insert(
            all_corpus_index_order_1D_pad_reshape[sample_ind][1].reshape(corpus_pad_sr.shape[-1], 1), 0,
            sample_ind % constant_batch_size, axis=1)
        input_set_X1.append(corpus_pad_sr[sample_ind][0])
        input_set_X2.append(corpus_pad_sr[sample_ind][1])
        input_set_X3.append(corpus_pad_sr[sample_ind][2])
        input_set_X4.append(preorder_sampled)  # index_preorder
        input_set_X5.append(postorder_sampled)  # index_postorder
    return [input_set_X1, input_set_X2, input_set_X3, input_set_X4, input_set_X5]


def create_model(EMBEDDING_DIM, constant_batch_size, embedding_matrix_source, embedding_matrix_target, word2ind_source, word2ind_tar,
                 max_length_sr, max_length_tr, vocab_size_target):
    inp_layer_normal = Input(shape=(None,), batch_size=constant_batch_size, name="normal_input")
    inp_layer_pre = Input(shape=(None,), batch_size=constant_batch_size, name="preorder_input")
    inp_layer_post = Input(shape=(None,), batch_size=constant_batch_size, name="postorder_input")
    inp_index_pre = Input(shape=(None, 2), batch_size=constant_batch_size, dtype=tf.int32, name="preorder_index")
    inp_index_post = Input(shape=(None, 2), batch_size=constant_batch_size, dtype=tf.int32, name="postorder_index")

    mid_layer_normal = Embedding(len(word2ind_source) + 1, EMBEDDING_DIM, weights=[embedding_matrix_source],
                                 input_length=max_length_sr, trainable=True)  # (inp_layer_normal)
    mid_layer_pre = Embedding(len(word2ind_source) + 1, EMBEDDING_DIM, weights=[embedding_matrix_source],
                              input_length=max_length_sr, trainable=True)  # (inp_layer_pre)
    mid_layer_post = Embedding(len(word2ind_source) + 1, EMBEDDING_DIM, weights=[embedding_matrix_source],
                               input_length=max_length_sr, trainable=True)  # (inp_layer_post)

    embd_normal = mid_layer_normal(inp_layer_normal)
    embd_pre = mid_layer_pre(inp_layer_pre)
    embd_post = mid_layer_post(inp_layer_post)

    mid_layer_normal = LSTM(units=EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embd_normal)
    mid_layer_pre = LSTM(units=EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embd_pre)
    mid_layer_post = LSTM(units=EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embd_post)

    # mid_layer_normal = Lambda(lambda x: tf.gather_nd(x[0],x[1]))([mid_layer_normal,inp_index_order])
    mid_layer_pre = Lambda(lambda x: tf.gather_nd(x[0], x[1]), name="reindex_preorder")([mid_layer_pre, inp_index_pre])
    mid_layer_post = Lambda(lambda x: tf.gather_nd(x[0], x[1]), name="reindex_postorder")(
        [mid_layer_post, inp_index_post])

    concat_layer = Concatenate(axis=2)([mid_layer_normal, mid_layer_pre, mid_layer_post])

    encoder_lstm = LSTM(units=EMBEDDING_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(concat_layer)
    # # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    #  #Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,), batch_size=constant_batch_size, name="decoder input")
    dec_emb_layer = Embedding(len(word2ind_tar) + 1, EMBEDDING_DIM, weights=[embedding_matrix_target],
                              input_length=max_length_tr, trainable=True)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_embedding = dec_emb_layer(decoder_inputs)  # not sharing the embedding layer
    decoder_lstm = LSTM(EMBEDDING_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(units=vocab_size_target, activation='softmax', name='softmax_layer')
    dense_time = tf.keras.layers.TimeDistributed(decoder_dense, name='time_distributed_layer')

    decoder_outputs = dense_time(decoder_outputs)

    # # Define the model that will turn
    # # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([inp_layer_normal, inp_layer_pre, inp_layer_post, inp_index_pre, inp_index_post, decoder_inputs],
                  decoder_outputs)

    # # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model


def shift(lst, vocab_size_h):
    new_list = []
    for sent in lst:
        sent_list = []
        for p in range(len(sent) - 1):
            sent_list.insert(p, sent[p + 1])
        z = np.zeros((vocab_size_h))
        z[0] = 1
        sent_list.append(z)
        new_list.append(sent_list)

    return new_list


def decode_sequence(input_seq, constant_batch_size, ind2word_tar, decoder_model, word2ind_tar,
                    encoder_model):
    indices = []
    target = []
    newtarget = []

    for j in range(constant_batch_size):
        target.append([])
        # print(j)
        stop_condition = False
        states_value = encoder_model.predict(input_seq, batch_size=constant_batch_size)

        target_seq = np.zeros((constant_batch_size, 1))
        target_seq[j, 0] = word2ind_tar['sos']
        k = 0

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, batch_size=constant_batch_size)
            integers = np.argmax(output_tokens[j])

            if integers == 0:
                word = "PADDING"
            else:
                integers = integers
                word = ind2word_tar[integers]
            target[j].append(word)
            if (word == 'eos' or len(target[j]) > 15):
                stop_condition = True

            if word is None:
                break
            #  # Update the target sequence (of length 1).
            target_seq = np.zeros((constant_batch_size, 1))
            target_seq[j, 0] = integers
            # Update states
            states_value = [h, c]

    for i in range(len(target)):
        newtarget.append([])
        for ele in target[i]:
            b = ele
            if b == 'PADDING':
                del (b)
            else:
                newtarget[i].append(b)

    return newtarget  # decode_sentence


def print_target(output1, ind2word_tar):
    indices = []
    target1 = []
    for j in range(len(output1)):
        target1.append([])
        for i in range(len(output1[-2])):
            integers = np.argmax(output1[j][i])
            if integers == 0:
                word = "PADDING"
            else:
                integers = integers
                word = ind2word_tar[integers]
            if word is None:
                break
            target1[j].append(word)

    print(target1)


def evaluate_model(eng, hin, target):
    smoother = SmoothingFunction()
    actual, predicted = list(), list()
    bb1 = 0
    bb2 = 0
    bb3 = 0
    bb4 = 0
    gleu_list = []
    # gleu1=0
    c = 0

    for i in range(128):
        raw_target, raw_src, tar = hin[i], eng[i], target[i]
        tar = tar[0:-1]
        raw_target = raw_target[1:-1]
        raw_src = raw_src[1:-1]
        # tar=[tar]
        # raw_target=[raw_target]
        # raw_src=raw_src[1:-1]
        if i < 128:
            print('src=%s, target=%s, pred=%s' % (raw_src, raw_target, tar))

        try:
            bb1 += (sentence_bleu([raw_target], tar, weights=(1, 0, 0, 0),
                                  smoothing_function=smoother.method2) + sentence_bleu(raw_target, tar,
                                                                                       weights=(1, 0, 0, 0),
                                                                                       smoothing_function=smoother.method7)) / 2
            print(bb1)
            # meteor+=nltk.translate.meteor_score.meteor_score([raw_target], "tar")
            bb2 += (sentence_bleu([raw_target], tar, weights=(0, 1, 0, 0),
                                  smoothing_function=smoother.method2) + sentence_bleu(raw_target, tar,
                                                                                       weights=(0, 1, 0, 0),
                                                                                       smoothing_function=smoother.method7)) / 2
            print(bb2)
            bb3 += (sentence_bleu([raw_target], tar, weights=(0, 0, 1, 0),
                                  smoothing_function=smoother.method2) + sentence_bleu(raw_target, tar,
                                                                                       weights=(0, 0, 1, 0),
                                                                                       smoothing_function=smoother.method7)) / 2
            print(bb3)
            bb4 += (sentence_bleu([raw_target], tar, weights=(0, 0, 0, 1),
                                  smoothing_function=smoother.method2) + sentence_bleu(raw_target, tar,
                                                                                       weights=(0, 0, 0, 1),
                                                                                       smoothing_function=smoother.method7)) / 2
            print(bb4)
            gleu1 = gleu.sentence_gleu([raw_target], tar, min_len=1, max_len=1)
            print(gleu1)
            gleu_list.append(gleu1)

        except ZeroDivisionError:
            pass

    bleu1 = bb1 * 100 / 128
    bleu2 = bb2 * 100 / 128
    bleu3 = bb3 * 100 / 128
    bleu4 = bb4 * 100 / 128
    gleu_avg = sum(gleu_list) / len(gleu_list)
    return bleu1, bleu2, bleu3, bleu4, gleu_avg


def try_bleu():
    references = ['सभी', 'वचन', 'निभायेंगे', ':', 'मंत्री', 'श्री', 'शर्मा']
    candidates = ['सभी', 'वचन', 'निभायेंगे', ':', 'मंत्री', 'श्री', 'शर्मा']
    score = corpus_bleu(references, candidates)
    print(score)

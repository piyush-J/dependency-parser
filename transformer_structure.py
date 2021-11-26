from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Lambda, Bidirectional, Attention
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.mha2 = layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)

        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model), ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask=None, padding_mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3


def create_model(num_encoder_tokens, num_decoder_tokens, max_source_length, max_target_length):
    EMBEDDING_DIM = 300  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 128  # Hidden layer size in feed forward network inside transformer

    inp_layer_normal = Input(shape=(None,), name="normal_input")
    inp_layer_pre = Input(shape=(None,), name="preorder_input")
    inp_layer_post = Input(shape=(None,), name="postorder_input")
    inp_index_pre = Input(shape=(None, 2), dtype=tf.int32, name="preorder_index")
    inp_index_post = Input(shape=(None, 2), dtype=tf.int32, name="postorder_index")

    mid_layer_normal = TokenAndPositionEmbedding(max_source_length, num_encoder_tokens + 1, EMBEDDING_DIM)
    mid_layer_pre = TokenAndPositionEmbedding(max_source_length, num_encoder_tokens + 1, EMBEDDING_DIM)
    mid_layer_post = TokenAndPositionEmbedding(max_source_length, num_encoder_tokens + 1, EMBEDDING_DIM)

    embd_normal = mid_layer_normal(inp_layer_normal)
    embd_pre = mid_layer_pre(inp_layer_pre)
    embd_post = mid_layer_post(inp_layer_post)
    print(embd_normal.shape)

    transformer_block_normal = TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)
    transformer_layer_pre = TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)
    transformer_layer_post = TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)

    mid_layer_normal = transformer_block_normal(embd_normal)
    mid_layer_pre = transformer_layer_pre(embd_pre)
    mid_layer_post = transformer_layer_post(embd_post)

    print("Before lambda: ", mid_layer_pre.shape, inp_index_pre.shape)
    # mid_layer_normal = Lambda(lambda x: tf.gather_nd(x[0],x[1]))([mid_layer_normal,inp_index_order])
    mid_layer_pre = Lambda(lambda x: tf.gather_nd(x[0], x[1]), name="reindex_preorder")([mid_layer_pre, inp_index_pre])
    mid_layer_post = Lambda(lambda x: tf.gather_nd(x[0], x[1]), name="reindex_postorder")(
        [mid_layer_post, inp_index_post])
    print(mid_layer_normal.shape, mid_layer_pre.shape, mid_layer_post.shape)

    concat_layer = Concatenate(axis=2)([mid_layer_normal, mid_layer_pre, mid_layer_post])
    concat_layer = layers.Dense(EMBEDDING_DIM * 2, activation="relu")(concat_layer)
    concat_layer = layers.Dense(EMBEDDING_DIM, activation="relu")(concat_layer)
    print("concat_layer: ", concat_layer)

    transformer_block_encoder = TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)
    encoder_outputs = transformer_block_encoder(concat_layer)

    transformer_layer_decoder = DecoderLayer(EMBEDDING_DIM, num_heads, ff_dim)
    decoder_inputs = Input(shape=(None,), name="decoder input")
    dec_emb_layer = TokenAndPositionEmbedding(max_target_length, num_encoder_tokens + 1, EMBEDDING_DIM)
    decoder_embedding = dec_emb_layer(decoder_inputs)  # not sharing the embedding layer
    print(decoder_embedding.shape, encoder_outputs.shape)
    decoder_outputs = transformer_layer_decoder(decoder_embedding, encoder_outputs)

    decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='softmax_layer')
    dense_time = tf.keras.layers.TimeDistributed(decoder_dense, name='time_distributed_layer')

    decoder_outputs = dense_time(decoder_outputs)

    model = Model([inp_layer_normal, inp_layer_pre, inp_layer_post, inp_index_pre, inp_index_post, decoder_inputs],
                  decoder_outputs)

    # # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    print(model.summary())
    # plot_model(model, to_file='/content/drive/My Drive/model.png')
    plot_model(model)

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization

def get_angles(pos, i, d):
    angles = pos / (10000 **(2*(i//2) /d))
    
    return angles

def positional_encoding(positions, d):
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], 
                            np.arange(d)[np.newaxis, :],
                            d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype = tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    
    return mask

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q. k.T)
    dk = k.shape[0]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    
    attention_weights = tf.keras.activations.softmax(scaled_attention_logits)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation = 'relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads = num_heads, key_dim = embedding_dim)
        self.ffn = FullyConnected(embedding_dim = embedding_dim, fully_connected_dim = fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon = layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon = layernorm_eps)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                 maximum_position_encoding, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads = num_heads,
                                        fully_connected_dim = fully_connected_dim,
                                        dropout_rate = dropout_rate,
                                        layernorm_eps = layernorm_eps)
                           for _ in range(self.num_layers)]
        self.droput = Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= np.sqrt(self.embedding_dim)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads = num_heads,
                                       key_dim = embedding_dim)
        self.mha2 = MultiHeadAttention(num_heads = num_heads,
                                       key_dim = embedding_dim)
        self.ffn = FullyConnected(embedding_dim = embedding_dim,
                                  fully_connected_dim = fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon = layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon = layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon = layernorm_eps)
        
        self.droput1 = Dropout(dropout_rate)
        self.droput2 = Dropout(dropout_rate)
        self.droput3 = Dropout(dropout_rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha(x, x, x, look_ahead_mask, return_attention_scores = True)
        attn1 = self.dropout(attn1, training)
        out1 = self.layernorm(x + attn1)
        
        attn2 = attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask, return_attention_scores = True)
        attn2 = self.dropout2(attn2, training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3, attn_weights_block1, attn_weights_block2
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
                 maximum_position_encoding, dropout_rate = 0.1, layernorm_eps = 1e-6):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)
        self.dec_layers = [DecoderLayer(embedding_dim = self.embedding_dim,
                                        num_heads = num_heads,
                                        fully_connected_dim = fully_connected_dim,
                                        dropout_rate = dropout_rate,
                                        layernorm_eps = layernorm_eps)
                           for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
        
        attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2_enc-dec_att'.format(i+1)] = block2
        
        return x, attention_weights
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
               target_vocab_size, max_positional_encoding_input,
               max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size, 
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)
    
        self.final_layer = Dense(target_vocab_size, activation = 'softmax')
    
    def call(self, input, target, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights

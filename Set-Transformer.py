#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
#%%
# Attention
class Attention(K.layers.Layer):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.wq = K.layers.Dense(d_model)
        self.wk = K.layers.Dense(d_model)
        self.wv = K.layers.Dense(d_model)   
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def attention(self, q, k, v):
        matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)
    
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
    
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
        attention_weights = tf.nn.softmax(tf.reduce_mean(scaled_attention_logits, axis=-1), axis=-1)[:, tf.newaxis, :]
        
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
    
    def call(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        self_attention, _ = self.attention(q, k, v)
        attn_output = self.layernorm(self_attention)
        
        return attn_output

def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
  
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    attention_weights =tf.nn.softmax(scaled_attention_logits, axis=-1)
  
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = K.layers.Dense(d_model, kernel_regularizer=K.regularizers.l2(0.01))
        self.wk = K.layers.Dense(d_model, kernel_regularizer=K.regularizers.l2(0.01))
        self.wv = K.layers.Dense(d_model, kernel_regularizer=K.regularizers.l2(0.01))
    
        self.dense = K.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x, y):
        batch_size = tf.shape(x)[0]
    
        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(y)  # (batch_size, seq_len, d_model)
        v = self.wv(y)  # (batch_size, seq_len, d_model)
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
    
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return q, output, attention_weights  
#%%
class MAB(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MAB, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = K.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = K.layers.LayerNormalization(epsilon=1e-6)
        
        self.rff = K.layers.Dense(d_model)

    def call(self, x, y):
        Q, MHA, _ = self.mha(x, y)
        H = self.layernorm1(Q + MHA)
        MAB = self.layernorm2(H + self.rff(H))
        
        return MAB
#%%
class SAB(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(SAB, self).__init__()
        self.mab = MAB(d_model, num_heads)
        
    def call(self, x):
        return self.mab(x, x)
#%%
class ISAB(K.layers.Layer):
    def __init__(self, m, d_model, num_heads):
        super(ISAB, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.I = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(1, m, d_model)))
        
        self.mab1 = MAB(d_model, num_heads)
        self.mab2 = MAB(d_model, num_heads)

    def call(self, x):
        H = self.mab1(self.I, x)
        ISAB = self.mab2(x, H)
        
        return ISAB
#%%
class PMA(K.layers.Layer):
    def __init__(self, d_model, num_heads, k=1):
        super(PMA, self).__init__()
        self.S = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(1, k, d_model)))
        self.mab = MAB(d_model, num_heads)
        self.rff = K.layers.Dense(d_model)
        
    def forward(self, x):
        return self.mab(self.S, self.rff(x))
#%%
class Encoder_ISAB(K.layers.Layer):
    def __init__(self, m, d_model, num_heads):
        super(Encoder_ISAB, self).__init__()
        self.isab1 = ISAB(m, d_model, num_heads)
        self.isab2 = ISAB(m, d_model, num_heads)
        
    def call(self, x):
        return self.isab2(self.isab1(x))
#%%
class Decoder(K.layers.Layer):
    def __init__(self, m, d_model, num_heads, k=1):
        super(Decoder, self).__init__()
        self.pma = PMA(d_model, num_heads, k=k)
        self.sab = SAB(d_model, num_heads)
        self.rff = K.layers.Dense(d_model)
        
    def call(self, x):
        return self.rff(self.sab(self.pma(x)))
    
#%%
class SetTransformer(K.models.Model):
    def __init__(self, m, d_model, num_heads, k=1):
        super(SetTransformer, self).__init__()
        self.encoder = Encoder_ISAB(m, d_model, num_heads)
        self.decoder = Decoder(m, d_model, num_heads, k=k)
    
    def call(self, x):
        return self.decoder(self.encoder(x))
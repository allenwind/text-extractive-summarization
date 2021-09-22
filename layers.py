import tensorflow as tf
from tensorflow.keras.layers import *

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

def gelu_tanh(x):
    """https://arxiv.org/abs/1606.08415"""
    cdf = 0.5 * (1.0 + tf.tanh(0.7978845608028654 * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf

def prior_softmax(inputs):
    x, prior, alpha = inputs

class MaskGlobalMaxPooling1D(Layer):
    """支持Mask的MaskGlobalMaxPooling1D"""
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        x = inputs
        # 用一个大的负数mask
        x = x - (1 - mask) * 1e12
        return tf.reduce_max(x, axis=1)

class MaskGlobalAveragePooling1D(Layer):
    """支持Mask的MaskGlobalAveragePooling1D"""

    def __init__(self, **kwargs):
        super(MaskGlobalAveragePooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        x = inputs
        x = tf.reduce_sum(x * mask, axis=1)
        return x / (tf.reduce_sum(mask, axis=1) + 1e-10)

class GlobalMasking(Layer):
    """计算全局的Masking"""

    def __init__(self, **kwargs):
        super(GlobalMasking, self).__init__(**kwargs)

    def call(self, inputs):
        mask = tf.not_equal(inputs, 0)
        mask = tf.expand_dims(mask, axis=2)
        # (batch_size, seq_len, 1)
        mask = tf.cast(mask, tf.float32)
        return mask

class PriorScaleShift(Layer):
    """学习输出贝叶斯先验分布"""

    def __init__(self, vocab_size, **kwargs):
        super(PriorScaleShift, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.log_scale = self.add_weight(
            name="log_scale",
            shape=(1, 1, self.vocab_size),
            initializer=tf.keras.initializers.zeros,
        )
        self.shift = self.add_weight(
            name="shift",
            shape=(1, 1, self.vocab_size),
            initializer=tf.keras.initializers.zeros,
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        # TODO：重点关注首尾文字
        x = inputs
        x = tf.one_hot(x, self.vocab_size)
        x = tf.reduce_sum(x * mask, axis=1, keepdims=True)
        x = tf.greater(x, 0)
        x = tf.cast(x, tf.float32)
        return tf.exp(self.log_scale) * x + self.shift

class MaskBiLSTM(Layer):
    """支持掩码的BiLSTM，比原生速度更快"""

    def __init__(self, hdims, **kwargs):
        super(MaskBiLSTM, self).__init__(**kwargs)
        self.hdims = hdims
        self.forward_lstm = LSTM(hdims, return_sequences=True)
        self.backend_lstm = LSTM(hdims, return_sequences=True)

    def reverse_sequence(self, x, mask):
        seq_len = tf.reduce_sum(mask, axis=1)[:, 0]
        seq_len = tf.cast(seq_len, tf.int32)
        x = tf.reverse_sequence(x, seq_len, seq_axis=1)
        return x

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        x = inputs
        x_forward = self.forward_lstm(x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.backend_lstm(x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = tf.concat([x_forward, x_backward], axis=-1)
        x = x * mask
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.hdims * 2,)

class FFN(Layer):
    """Transformer中FNN的一种形式"""

    def __init__(self, output_dim, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_1 = Dense(output_dim)
        self.dense_2 = Dense(output_dim)
        self.dense_3 = Dense(output_dim)

    def call(self, inputs, mask=None):
        x = inputs
        w1 = self.dense_1(x)
        w2 = self.dense_2(x)
        x = gelu(w1) * w2
        x = self.dense_3(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

class ConditionalLayerNormalization(Layer):
    """参考：https://arxiv.org/pdf/1810.01365"""

    def __init__(self, hdims, **kwargs):
        super(ConditionalLayerNormalization, self).__init__(**kwargs)
        self.hdims = hdims

    def build(self, input_shape):
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta1 = Dense(self.hdims, activation="relu")
        self.beta2 = Dense(output_dim)
        self.gamma1 = Dense(self.hdims, activation="relu")
        self.gamma2 = Dense(output_dim)

    def call(self, inputs):
        x, cond = inputs
        x = self.layernorm(x)
        beta = self.beta1(cond)
        beta = self.beta2(beta)
        gamma = self.gamma1(cond)
        gamma = self.gamma2(gamma)
        beta = tf.expand_dims(beta, axis=1)
        gamma = tf.expand_dims(gamma, axis=1)
        return x * (gamma + 1.0) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Attention(Layer):

    def __init__(
        self,
        heads,
        size_per_head,
        use_scale=True,
        key_size=None,
        return_attention_weights=False,
        **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.use_scale = use_scale
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        if self.use_scale:
            self.scale = self.key_size ** 0.5
        else:
            self.scale = 1.0
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
        self.dropout = Dropout(0.1)

    def call(self, inputs, mask=None, training=None):
        q, k, v = inputs
        v_mask = mask

        # QKV的线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = tf.reshape(qw, (-1, tf.shape(qw)[1], self.heads, self.key_size))
        kw = tf.reshape(kw, (-1, tf.shape(kw)[1], self.heads, self.key_size))
        vw = tf.reshape(vw, (-1, tf.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = tf.transpose(qw, (0, 2, 1, 3))
        kw = tf.transpose(kw, (0, 2, 1, 3))
        vw = tf.transpose(vw, (0, 2, 1, 3))
        # Attention
        a = tf.einsum("ijkl,ijml->ijkm", qw, kw) / self.scale
        a = self.dropout(a, training=training)
        a = tf.transpose(a, (0, 3, 2, 1))

        v_mask = tf.expand_dims(v_mask, axis=-1)
        # padding部分减去一个大数
        a = a - (1 - v_mask) * 1e12

        a = tf.transpose(a, (0, 3, 2, 1))
        a = tf.math.softmax(a, axis=-1)

        # 完成输出
        o = tf.einsum("ijkl,ijlm->ijkm", a, vw)
        o = tf.transpose(o, (0, 2, 1, 3))
        o = tf.reshape(o, (-1, tf.shape(o)[1], self.out_dim))
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

class SparseCrossEntropy(Layer):

    def __init__(self, **kwargs):
        super(SparseCrossEntropy, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        y_true, y_pred = inputs
        loss = self.compute_loss(y_true, y_pred)
        self.add_loss(loss)
        return y_pred

    def compute_loss(self, y_true, y_pred):
        y_mask = tf.cast(tf.expand_dims(tf.not_equal(y_true, 0), axis=2), tf.float32)
        # Teacher-Forcing对输入和预测错开一位
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:, 0]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(loss * y_mask) / tf.reduce_sum(y_mask)
        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class AutoWeightAverage(Layer):
    """参数化的加权平均"""

    def __init__(self, **kwargs):
        super(AutoWeightAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ws = self.add_weight(
            name="weights",
            shape=(2,),
            initializer=tf.keras.initializers.ones,
            trainable=True,
        )

    def call(self, inputs, mask=None):
        xy, x_prior = inputs
        x = xy * self.ws[0] + x_prior * self.ws[1]
        return x / tf.reduce_sum(self.ws)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])

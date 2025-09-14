""" 
Transformer layers for EEG-ATCNet enhancement
Implementing Convolutional-Attention Hybrid Model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Add, Lambda
from tensorflow.keras import backend as K

class PositionalEncoding(Layer):
    """位置编码层，为EEG时序数据添加位置信息"""
    
    def __init__(self, max_seq_length=1000, d_model=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
    def build(self, input_shape):
        # 创建位置编码矩阵
        position = np.arange(self.max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.max_seq_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.Variable(
            initial_value=pos_encoding.astype(np.float32),
            trainable=False,
            name='positional_encoding'
        )
        
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_length, :]
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model,
        })
        return config

class TransformerEncoderBlock(Layer):
    """Transformer编码器块"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Feed forward network
        self.ffn1 = Dense(dff, activation='relu')
        self.ffn2 = Dense(d_model)
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        # Multi-head attention with residual connection
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network with residual connection
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        })
        return config

def create_transformer_encoder(d_model=128, num_heads=4, num_layers=2, dff=256, dropout_rate=0.1):
    """
    创建Transformer编码器
    
    Parameters:
    -----------
    d_model : int
        模型维度
    num_heads : int
        注意力头数
    num_layers : int
        编码器层数
    dff : int
        前馈网络维度
    dropout_rate : float
        Dropout率
    """
    
    def transformer_encoder(inputs):
        x = inputs
        
        # 堆叠多层Transformer编码器
        for i in range(num_layers):
            x = TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i+1}'
            )(x)
        
        return x
    
    return transformer_encoder

def eeg_sequence_pooling(transformer_output, pooling_method='cls_token'):
    """
    对Transformer输出进行序列池化
    
    Parameters:
    -----------
    transformer_output : tensor
        Transformer输出 (batch, seq_length, d_model)
    pooling_method : str
        池化方法：'cls_token', 'mean', 'max', 'attention'
    """
    
    if pooling_method == 'cls_token':
        # 使用第一个token（类似BERT的[CLS]）
        return Lambda(lambda x: x[:, 0, :], name='cls_token_pooling')(transformer_output)
    
    elif pooling_method == 'mean':
        # 全局平均池化
        return Lambda(lambda x: tf.reduce_mean(x, axis=1), name='mean_pooling')(transformer_output)
    
    elif pooling_method == 'max':
        # 全局最大池化
        return Lambda(lambda x: tf.reduce_max(x, axis=1), name='max_pooling')(transformer_output)
    
    elif pooling_method == 'attention':
        # 注意力池化
        attention_weights = Dense(1, activation='softmax', name='attention_pooling_weights')(transformer_output)
        pooled = Lambda(lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=1), 
                       name='attention_pooling')([transformer_output, attention_weights])
        return pooled
    
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

def prepare_cnn_features_for_transformer(cnn_features, target_d_model=128):
    """
    将CNN特征准备为Transformer输入格式
    
    Parameters:
    -----------
    cnn_features : tensor
        CNN输出特征，形状可能是 (batch, features) 或 (batch, seq, features)
    target_d_model : int
        目标Transformer模型维度
    """
    
    # 如果是2D特征，需要扩展为序列
    if len(cnn_features.shape) == 2:
        # (batch, features) → (batch, 1, features)
        cnn_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(cnn_features)
    
    # 投影到目标维度
    if cnn_features.shape[-1] != target_d_model:
        cnn_features = Dense(target_d_model, name='feature_projection')(cnn_features)
    
    return cnn_features
